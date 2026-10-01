"""C30 headless task: original physical limits and one fixed 400-control retention."""
from __future__ import annotations

import math
import numpy as np
from reader26 import require, close, _ground_hit
from side_math27 import scalar_state, compiled_side_geometry, support_margin
from score27 import native_loads, stopped_window


def score_skill30(episode, binding, geometry, kin, case_id, direction, original_spec):
    rows, states, native = episode['rows'], episode['states'], episode['native']
    gates = original_spec['gates']
    side = [i for i, row in enumerate(rows) if row['actor'] == 'traditional_side']
    if not side:
        return {'base_task_gates_passed': False, 'side_started': False,
                'reason': 'no admitted side interval; no skill success claimed'}
    start, end = side[0], side[-1]+1
    require(side == list(range(start, end)), 'C30 side controls are not one contiguous interval')
    first = rows[start]['info']['controller_record']
    last = rows[end-1]['info']['controller_record']
    finish = last['diagnostic_after']
    q0 = states['qpos'][start]
    st0 = scalar_state(q0, states['qvel'][start], binding)
    yaw0 = st0['rpy'][2]
    fwd = np.array([math.cos(yaw0), math.sin(yaw0)])
    left = np.array([-math.sin(yaw0), math.cos(yaw0)])
    displacement = states['qpos'][:, :2]-q0[:2]
    signed, longitudinal = direction*(displacement@left), displacement@fwd
    target = q0[:2]+direction*.03*left
    require([i for i in side if rows[i]['info']['controller_record']['accepted_start']] == [start]
            and first['intent_direction'] == direction
            and close(first['diagnostic_before']['initial_pose'], q0[:3])
            and close(first['diagnostic_before']['delta'], np.r_[direction*.03*left, 0.]),
            'C30 actual start/goal differs from fixed 30 mm task')
    loads, contacts = native_loads(native, gates['positive_native_load_threshold_N'])
    beginning = compiled_side_geometry(kin, q0, states['qvel'][start], binding['wheel_map'])
    ending = compiled_side_geometry(kin, states['qpos'][end], states['qvel'][end], binding['wheel_map'])
    margin = support_margin(ending['contact_points_m'][:, :2], ending['whole_com_m'][:2])
    entry = {'exact_200_preparation': start == 200,
             'roll_pitch': bool(np.max(np.abs(st0['rpy'][:2])) <= gates['entry_abs_roll_pitch_le_rad']),
             'origin_speed': bool(np.linalg.norm(st0['origin_velocity']) < gates['entry_origin_speed_lt_mps']),
             'angular_speed': bool(np.linalg.norm(st0['world_angular_velocity']) < gates['entry_angvel_norm_lt_rps']),
             'foot_plane': bool(np.max(np.abs(beginning['contact_points_m'][:, 2])) <= gates['entry_foot_plane_abs_le_m']),
             'four_actual_contacts': bool(start > 0 and contacts[5*start-1].all())}
    liftoff, touchdown = [False]*4, [False]*4
    for i in side:
        shaped = compiled_side_geometry(kin, states['qpos'][i+1], states['qvel'][i+1], binding['wheel_map'])
        for wheel in range(4):
            touchdown[wheel] |= bool(liftoff[wheel] and loads[5*i:5*i+5, wheel].any())
            liftoff[wheel] |= bool(shaped['contact_points_m'][wheel, 2] > .012
                                  and shaped['wheel_shape_min_z_m'][wheel] > 0.
                                  and not contacts[5*i+4, wheel])
    leg_dofs = np.asarray(binding['dof_addresses']).reshape(4, 4)[:, :3].ravel()
    side_safe = all(abs(r['roll_deg']) <= math.degrees(.32) and abs(r['pitch_deg']) <= math.degrees(.32)
                    and r['after']['qpos'][2] >= .32 and r['nonwheel_contact_count'] == 0
                    and np.max(np.abs(np.asarray(r['after']['qvel'])[leg_dofs])) <= 18.
                    for r in native[5*start:5*end])
    rolling_safe = all(abs(r['roll_deg']) <= 10 and abs(r['pitch_deg']) <= 10
                       and r['after']['qpos'][2]-_ground_hit(geometry, *r['after']['qpos'][:2])[0] >= .28
                       and r['nonwheel_contact_count'] == 0 and abs(r['after']['qpos'][0]) <= 11
                       and abs(r['after']['qpos'][1]) <= 6
                       for i, row in enumerate(rows) if row['actor'] != 'traditional_side'
                       for r in native[5*i:5*i+5])
    complete = bool(finish['done'] and finish['leg'] is None and last['handoff_next_preview'])
    landed = bool(complete and contacts[5*end-1].all() and loads[5*end-1].all()
                  and margin >= gates['finish_margin_ge_m'])
    position_error = float(np.linalg.norm(states['qpos'][end, :2]-target))
    yaw_end = scalar_state(states['qpos'][end], states['qvel'][end], binding)['rpy'][2]
    yaw_error = math.atan2(math.sin(yaw_end-yaw0), math.cos(yaw_end-yaw0))
    is_cancel = case_id.startswith('cancel_')
    cancel_rows = [i for i in side if rows[i]['info']['controller_record']['cancel_requested']]
    cancel_ok = bool(cancel_rows and finish['failure'] == 'cancelled' and not finish['status']['success']
                     and 0 < end-cancel_rows[0] <= 300)
    success = bool(not is_cancel and finish['failure'] is None and finish['status']['success']
                   and gates['signed_lateral_m'][0] <= signed[end] <= gates['signed_lateral_m'][1]
                   and position_error < gates['final_xy_error_lt_m']
                   and abs(yaw_error) < gates['final_yaw_error_lt_rad'] and all(liftoff) and all(touchdown))
    observation_end = end+400
    complete_retention = complete and observation_end == len(rows)
    retention = rollback = None
    if observation_end <= len(rows):
        retention = float(signed[observation_end]/signed[end]) if signed[end] != 0 else None
        rollback = float(signed[end]-signed[end:observation_end+1].min())
    retained = is_cancel or (retention is not None and retention >= gates['retention_ge']
                            and rollback <= gates['max_rollback_le_m'])
    stopped = stopped_window(rows, states, loads, observation_end-100, observation_end,
                             binding, geometry, gates)
    all_zero = all(all(row['info'][key][field] == 0.
                       for key in ('raw_operator_command', 'consumed_command')
                       for field in ('forward_velocity_mps', 'lateral_velocity_mps', 'yaw_rate_rps')) for row in rows)
    resumed = bool(complete_retention and all(row['actor'] == 'B' and row['policy_predict_called'] is True
                   for row in rows[end:observation_end]))
    checks = {'entry': all(entry.values()), 'side_physical_safety': bool(side_safe),
              'rolling_physical_safety': bool(rolling_safe), 'no_environment_termination': not any(r['terminated'] for r in rows),
              'all_raw_servo_motion_zero': all_zero, 'side_controls_within_1500': end-start <= 1500,
              'cycle_within_original_15s': (end-start)*.01 <= gates['cycle_limit_s'],
              'safe_landed_handoff': landed, 'cancel_safe_or_complete_step': cancel_ok if is_cancel else success,
              'exact_400_retention_complete': bool(complete_retention), 'retention_if_success': bool(retained),
              'original_tail_stopped_and_loaded': stopped['passed'], 'B22_predict_restored_in_retention': resumed}
    return {'base_task_gates_passed': all(checks.values()), 'gates': checks, 'entry_gates': entry,
            'side_started': True, 'case_id': case_id, 'is_cancellation_case': is_cancel,
            'side_control_window': [start, end], 'side_duration_s': (end-start)*.01,
            'retention_control_window': [end, observation_end],
            'signed_lateral_m': float(signed[end]), 'final_xy_error_m': position_error,
            'final_yaw_error_rad': yaw_error, 'finish_support_margin_m': float(margin),
            'nominal_30mm_per_cycle_mps': .03/((end-start)*.01) if not is_cancel else None,
            'actual_net_per_cycle_mps': float(signed[end]/((end-start)*.01)) if not is_cancel else None,
            'max_abs_longitudinal_drift_m': float(np.abs(longitudinal[start:end+1]).max()),
            'each_wheel_liftoff': liftoff, 'each_wheel_loaded_touchdown': touchdown,
            'retention_ratio': retention, 'max_rollback_m': rollback, 'tail_stop': stopped,
            'cancel_to_handoff_controls': end-cancel_rows[0] if cancel_rows else None,
            'new_reference_geometry_and_macro_gates_checked_separately': True,
            'full_skill_qualification_not_decided_by_this_base_score_alone': True}
