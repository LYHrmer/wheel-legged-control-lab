"""C30 independent leg-action, saved shape gate and analytic-reference checks."""
from __future__ import annotations

import math
import numpy as np
from reader26 import require, close
from side_math27 import compiled_side_geometry, support_margin
from geometry_read30 import wheel_envelope30, effective_ground_contact30

SCHEMA = 'd1-c30-lateral-reference-skill-v1'


def curve30(t, duration):
    u = min(1., max(0., t/duration))
    if u == 0. or u == 1.:
        return u, 0., 0.
    return (u**3*(10.-15.*u+6.*u*u), 30.*u*u*(1.-u)**2/duration,
            60.*u*(1.-u)*(1.-2.*u)/duration**2)


def verify_latches30(episode, events, kin, binding, limits, case_id):
    rows, states = episode['rows'], episode['states']
    direction = 1 if case_id.endswith('_left') else -1
    order = [2, 0, 3, 1] if direction == 1 else [3, 1, 2, 0]
    flattened = [event for row in rows for event in row['macro_events30']]
    require(flattened == events and 1 <= len(events) <= 4, 'saved macro latches differ from actual control rows')
    summaries = []
    for mi, event in enumerate(events):
        i = event['control_index']
        row = rows[i]
        record = row['info']['controller_record']
        before, after = record['diagnostic_before'], record['diagnostic_after']
        require(event['schema'] == SCHEMA and event['macro_index'] == mi
                and event['latch_control_index'] == i and event['leg_index'] == order[mi]
                and row['actor'] == 'traditional_side' and event in row['macro_events30'],
                'latch identity/order does not name its actual consumed side control')
        original_from, original_to = (np.asarray(event[key], dtype=float)
                                     for key in ('teacher_body_from_m', 'teacher_body_to_m'))
        require(original_from.shape == original_to.shape == (3,)
                and close(original_from, before['pose'])
                and close(event['consumed_body_from_m'], original_from)
                and close(event['initial_body_yaw_rad'], before['start_yaw_rad']),
                'leg plan did not begin at the actual controller pose and initial heading')
        yaw = float(event['initial_body_yaw_rad'])
        rotate = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        if case_id.startswith('zero_'):
            proposed = projected = consumed = np.zeros(3)
            reason = 'teacher_exact_zero'
            require(event['extra_plan_candidate_qpos23'] is None and event['extra_plan_ik_error_m'] is None,
                    'zero action executed a new plan IK')
        else:
            toward = original_from[:2]-original_to[:2]
            distance = np.linalg.norm(toward)
            delta = np.zeros(2) if distance == 0 else min(.008, distance)*toward/distance
            proposed = np.r_[rotate.T@delta, .5]
            projected = np.r_[np.clip(proposed[:2], -.012, .012), .5]
            candidate = original_to.copy()
            candidate[:2] += rotate@projected[:2]
            consumed = np.array([0., 0., .5])
            if np.linalg.norm(candidate[:2]-original_from[:2]) > .120:
                reason = 'max_shift_rejected_xy'
                require(event['extra_plan_candidate_qpos23'] is None, 'rejected span performed extra IK')
            else:
                q = np.asarray(event['extra_plan_candidate_qpos23'], dtype=float)
                require(q.shape == (23,) and np.isfinite(q).all() and close(q[:3], candidate)
                        and close(q[3:7], [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]),
                        'candidate IK result is not the declared body reference')
                g = compiled_side_geometry(kin, q, np.zeros(22), binding['wheel_map'])
                stance = [k for k in range(4) if k != order[mi]]
                margin = support_margin(g['contact_points_m'][stance, :2], g['whole_com_m'][:2])
                require(close(g['contact_points_m'], event['extra_plan_contact_points_m'], atol=2e-9)
                        and close(g['whole_com_m'], event['extra_plan_com_m'], atol=2e-9)
                        and close(margin, event['predicted_tripod_static_margin_m'], atol=2e-9),
                        'candidate IK contact geometry/COM/margin is not its actual saved qpos')
                qa = np.asarray(binding['qpos_addresses'], dtype=int)
                joint_ids = np.asarray([list(kin['jnt_qposadr']).index(int(address)) for address in qa], dtype=int)
                bounds = np.asarray(limits['jnt_range'], dtype=float)[joint_ids]
                limited = np.asarray(limits['jnt_limited'], dtype=bool)[joint_ids]
                in_range = bool(np.all(q[qa][limited] >= bounds[limited, 0]-1e-12)
                                and np.all(q[qa][limited] <= bounds[limited, 1]+1e-12))
                err = event['extra_plan_ik_error_m']
                accepted = math.isfinite(err) and err <= .008 and margin >= .028 and in_range
                reason = 'body_xy_projected_and_consumed' if accepted else 'extra_plan_IK_or_tripod_margin_rejected_xy'
                if accepted:
                    consumed = projected.copy()
        require(close(event['proposed_action'], proposed) and close(event['projected_action'], projected)
                and close(event['consumed_action'], consumed) and event['reason'] == reason,
                'leg action/projection does not follow the one preregistered fixed candidate')
        expected_to = original_to.copy()
        expected_to[:2] += rotate@consumed[:2]
        require(close(event['consumed_body_to_m'], expected_to) and close(after['body_to'], expected_to),
                'actual consumed body waypoint differs from its latched action')
        g = compiled_side_geometry(kin, states['qpos'][i], states['qvel'][i], binding['wheel_map'])
        height = .45 if mi == 0 else float(np.clip(g['whole_com_m'][2]-g['contact_points_m'][:, 2].min(), .10, 1.))
        require(close(event['plan_com_height_m'], height), 'shift acceleration used a different actual COM height')
        def move_time(target):
            return max(.30, math.sqrt(max(1e-9, 5.7735*np.linalg.norm(target[:2]-original_from[:2])*height/(9.81*.015))))
        require(close(event['teacher_shift_time_s'], move_time(original_to))
                and close(event['consumed_shift_time_s'], move_time(expected_to))
                and close(after['shift_time_s'], event['consumed_shift_time_s']),
                'new shift time violates the original ZMP bound or invents an upper-time clip')
        summaries.append({'macro_index': mi, 'control_index': i, 'leg_index': order[mi],
                          'action': consumed.tolist(), 'projection_reason': reason})
    return summaries


def native_geometry30(episode, kin, binding):
    minima, contacts = [], []
    for native in episode['native']:
        proof = native['skill30_geometry']
        actual = wheel_envelope30(kin, native['after']['qpos'], native['after']['qvel'], binding['wheel_map'])
        active = [effective_ground_contact30(native, leg) for leg in range(4)]
        require(proof['native_index'] == native['native_index']
                and proof['start_time_s'] == native['start_time_s'] and proof['end_time_s'] == native['end_time_s']
                and close(proof['whole_wheel_min_z_m'], actual['wheel_shape_min_z_m'], atol=2e-9)
                and proof['active_wheel_terrain_contact'] == active,
                'saved native whole-wheel geometry/contact proof differs from actual state/contacts')
        minima.append(actual['wheel_shape_min_z_m'])
        contacts.append(active)
    return np.asarray(minima), np.asarray(contacts, dtype=bool)


def verify_reference30(episode, events, kin, binding, case_id, native_minima, native_contacts):
    rows, states = episode['rows'], episode['states']
    zero = case_id.startswith('zero_')
    overlaps = {i: 0 for i in range(len(events))}
    geometry_checks = []
    previous_by_macro = {}
    for i, row in enumerate(rows):
        if row['actor'] != 'traditional_side':
            require(row['skill30'] is None and not row['macro_events30'], 'rolling row invented a skill event')
            continue
        rec, skill = row['info']['controller_record'], row['skill30']
        before, after = rec['diagnostic_before'], rec['diagnostic_after']
        mi = max(j for j, e in enumerate(events) if e['control_index'] <= i)
        event = events[mi]
        require(skill['schema'] == SCHEMA and skill['control_index'] == i and skill['macro_index'] == mi
                and skill['phase'] == after['phase'] and skill['leg_index'] == after['leg']
                and skill['early_lower_enabled'] is False
                and skill['early_lower_reason'] == 'no_future_reference_shape_certificate'
                and close(skill['body_reference_position_m'], after['pose'])
                and close(skill['body_reference_accel_mps2'], after['body_accel'])
                and close(skill['foot_offset_m'], after['foot_offset']), 'skill reference not joined to actual controller state')
        phase, leg = after['phase'], after['leg']
        duration = float(after['swing_time_s'])
        require(close(duration, np.clip(1.875*.03/.16, .22, .90)), 'original foot horizontal speed/time changed')
        if leg is not None:
            require(close(skill['foot_reference_position_m'], np.asarray(after['feet'])[leg]+after['foot_offset'])
                    and skill['foot_reference_position_source'] == 'feet_plus_foot_offset_before_IK_height_correction',
                    'analytic pre-IK foot reference mislabeled as actual collision geometry')
        else:
            require(skill['foot_reference_position_m'] is None, 'no-leg phase invented a moving foot')
        if zero:
            require(skill['reference_qpos23'] is None and skill['reference_whole_wheel_min_z_m'] is None
                    and close(event['consumed_action'], [0., 0., 0.]), 'zero path evaluated nonzero reference')
            continue
        reference = np.asarray(skill['reference_qpos23'], dtype=float)
        require(reference.shape == (23,)
                and close(reference[np.asarray(binding['qpos_addresses'])], rec['side_calculation']['target_position_rad']),
                'saved reference configuration is not the actual consumed IK target')
        refshape = wheel_envelope30(kin, reference, np.zeros(22), binding['wheel_map'])['wheel_shape_min_z_m']
        require(close(skill['reference_whole_wheel_min_z_m'], refshape, atol=2e-9),
                'post-IK reference whole-wheel shape differs from its saved compiled configuration')
        start = skill['horizontal_start_control_index']
        elapsed = float(skill['horizontal_elapsed_s'])
        previous = previous_by_macro.get(mi)
        healthy_air = before['phase'] in ('lift', 'swing') and after['failure'] is None
        if start is not None:
            require(type(start) is int and event['control_index'] <= start <= i,
                    'horizontal clock starts before actual leg latch or in the future')
            if previous is not None and previous['horizontal_start_control_index'] is not None:
                require(start == previous['horizontal_start_control_index'], 'horizontal clock was restarted')
                expected = previous['horizontal_elapsed_s']+(.01 if healthy_air else 0.)
                require(close(elapsed, expected), 'horizontal elapsed time is not actual consumed controls')
            elif start == i:
                require(elapsed == 0., 'horizontal start must preserve zero position/velocity at this boundary')
            else:
                require(False, 'horizontal start has no recorded trigger boundary')
        else:
            require(elapsed == 0., 'horizontal elapsed before its event trigger')
        require(skill['horizontal_active'] is bool(start is not None and elapsed < duration),
                'horizontal-active label differs from actual reference clock')
        if healthy_air:
            oldleg = before['leg']
            gate = skill['gate']
            actual = wheel_envelope30(kin, states['qpos'][i], states['qvel'][i], binding['wheel_map'])
            free = not bool(native_contacts[5*(i-1):5*i, oldleg].any())
            require(i > 0 and gate['previous_interval_index'] == i-1
                    and gate['previous_contact_free_5'] is free
                    and close(gate['actual_whole_wheel_min_z_m'], actual['wheel_shape_min_z_m'][oldleg], atol=2e-9)
                    and gate['early_lower_reason'] == 'no_future_reference_shape_certificate',
                    'horizontal permission used future/stale/scratch evidence')
            if start == i:
                t = before['phase_time_s']+.01
                early = t >= (1-event['consumed_action'][2])*.26
                support = bool(after['dynamic_margin_m'] > .020
                               and all(after['wheel_contacts'][j] for j in range(4) if j != oldleg))
                require(actual['wheel_shape_min_z_m'][oldleg] > .012 and free
                        and (t >= .26 or early and support), 'horizontal start bypassed actual clearance/support gate')
        expected_velocity = np.zeros(3)
        expected_accel = np.zeros(3)
        if phase in ('lift', 'swing'):
            if start is not None:
                f, v, a = curve30(elapsed, duration)
                require(close(np.asarray(after['foot_offset'])[:2], np.asarray(after['delta'])[:2]*f),
                        'horizontal foot target differs from its one latched quintic')
                expected_velocity[:2] = np.asarray(after['delta'])[:2]*v
                expected_accel[:2] = np.asarray(after['delta'])[:2]*a
            if phase == 'lift':
                f, v, a = curve30(after['phase_time_s'], .26)
                require(close(after['foot_offset'][2], .035*f), 'lift height/clock differs')
                expected_velocity[2], expected_accel[2] = .035*v, .035*a
            else:
                require(close(after['foot_offset'][2], .035), 'swing reference began lowering early')
        elif phase == 'lower' and leg is not None:
            t = after['phase_time_s']
            require(elapsed >= duration, 'lower began before horizontal completion')
            require(close(np.asarray(after['foot_offset'])[:2], np.asarray(after['delta'])[:2]), 'lower changed lateral goal')
            if t <= .26:
                f, v, a = curve30(t, .26)
                require(close(after['foot_offset'][2], .035-(.035-.006)*f), 'serial lower height differs')
                expected_velocity[2], expected_accel[2] = -(.035-.006)*v, -(.035-.006)*a
            elif after['foot_offset'][2] > -.015:
                expected_velocity[2] = -.030
        require(close(skill['foot_reference_velocity_mps'], expected_velocity)
                and close(skill['foot_reference_accel_mps2'], expected_accel),
                'saved foot derivative is not its actual phase reference')
        moving = bool(start is not None and healthy_air and
                      (elapsed < duration or previous is not None and previous['horizontal_elapsed_s'] < duration))
        if moving:
            active_leg = before['leg']
            clear = bool(np.all(native_minima[5*i:5*i+5, active_leg] > .012))
            free = bool(not native_contacts[5*i:5*i+5, active_leg].any())
            geometry_checks.append({'control_index': i, 'leg_index': active_leg,
                                    'all_5T_whole_wheel_above_12mm': clear, 'all_5T_contact_free': free})
        if np.linalg.norm(expected_velocity[:2]) > 1e-10 and abs(expected_velocity[2]) > 1e-10:
            overlaps[mi] += 1
        previous_by_macro[mi] = skill
    return {'reference_arithmetic_verified': True,
            'horizontal_native_geometry_passed': all(r['all_5T_whole_wheel_above_12mm'] and r['all_5T_contact_free']
                                                      for r in geometry_checks),
            'horizontal_native_control_checks': geometry_checks,
            'overlap_controls_per_macro': overlaps,
            'actual_overlap_legs': sum(n >= 1 for n in overlaps.values()),
            'early_lower_enabled': False, 'reference_target_shape_does_not_retroactively_authorize_horizontal_start': True}
