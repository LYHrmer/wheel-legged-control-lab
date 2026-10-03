"""C34 saved physical proof: preserve C31/C27 chain with explicit new inputs.

Function globals are rebound on local code objects; sealed reader modules and
records are never mutated. The C27 all-row/native check still runs in full.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

import episode_read31 as inherited
import records27
from reader26 import require, close, saved_document
from reference_read30 import native_geometry30
from side_math27 import LIMITS
from force_read34 import recompute_side_torque34
from latches_read33 import verify_latches33
from reference_read33 import verify_reference33
from reference_read34 import verify_reference34
from score_read34 import score_skill34
from reward_read34 import verify_macro_receipts34


def next_side_index34(last_index, actual_index):
    """C27 records27 increments side_count before checking the saved index."""
    if type(actual_index) is not int or actual_index != last_index+1:
        raise ValueError('C34 force proof lost a 1-based side control or changed order')
    return actual_index


def physical_episode34(run, session, construction, binding, geometry, kin,
                       original_spec, cycle_index, offset, *, beta34,
                       distance_m):
    if beta34 not in (0., .5, 1.) or distance_m not in (.03, .04):
        raise ValueError('C34 reader case beta/distance differs')
    # C27 increments its side_count before comparing side_compute_index;
    # the first real side compute is 1, not 0.
    history = {'cancelled': False, 'last_index': 0}

    def force(record, qpos, qvel, original_binding, original_kin, profile):
        index = int(record['side_compute_index'])
        history['last_index'] = next_side_index34(history['last_index'], index)
        history['cancelled'] |= bool(record['cancel_requested'])
        return recompute_side_torque34(
            record, qpos, qvel, original_binding, original_kin, profile,
            cancelled_history=history['cancelled'], beta34=beta34,
            distance_m=distance_m)

    check = FunctionType(records27.check_episode.__code__,
        dict(records27.check_episode.__globals__, recompute_side_torque=force),
        'C34_C27_all_rows_native_and_new_wrench')
    local = FunctionType(inherited.local_episode31.__code__,
        dict(inherited.local_episode31.__globals__,
             records27=SimpleNamespace(check_episode=check)),
        'C34_C31_local_episode_with_complete_C27_force_proof')

    def latches(*args, **kwargs):
        return verify_latches33(*args, **kwargs, alpha33=1.)

    def original_foot(*args, **kwargs):
        return verify_reference34(*args, **kwargs, distance_m=distance_m)

    body_reference = FunctionType(verify_reference33.__code__,
        dict(verify_reference33.__globals__, verify_reference30=original_foot),
        'C34_C33_body_reference_with_actual_distance_foot')

    def reference(*args, **kwargs):
        return body_reference(*args, **kwargs, alpha33=1.)

    episode = local(run, session, construction, binding, geometry, kin,
                    cycle_index, offset)
    cycle = episode['cycle_receipt31']
    folder = Path(run)/f'episode_{cycle_index}'
    saved = saved_document(folder/'macro_transitions31.json')
    events = saved['events']
    mode, direction = cycle['mode'], cycle['direction']
    require(saved['schema'] == 'd1-c34-actual-event-macros-v1'
            and saved['elapsed_controls'] == episode['controls']
            and saved['global_control_offset'] == offset
            and saved['direction'] == direction and mode == 'fixed',
            'C34 actual-D macro episode identity differs')
    latch_report = latches(episode, events, construction, binding,
                           cycle_index=cycle_index, direction=direction,
                           mode=mode, rollout_parameters=None)
    minima, contacts = native_geometry30(episode, kin, binding)
    is_cancel = bool(cycle['result']['cancel_control_index'] is not None)
    case_id = cycle['case_id']
    require(case_id.startswith('cancel_') is is_cancel,
            'C34 cancel case label differs from real cancel control')
    reference_report = reference(episode, events, kin, binding, case_id,
                                 minima, contacts)
    task_report = score_skill34(episode, binding, geometry, kin, case_id,
                                direction, original_spec, distance_m=distance_m)
    side_rows = [r for r in episode['rows'] if r['actor'] == 'traditional_side']
    finish = side_rows[-1]['info']['controller_record']['diagnostic_after']
    end = side_rows[-1]['control_index']+1
    retained = finish['done'] and episode['controls'] == end+400
    teacher_reason = ('safe_cancel' if retained and finish['failure'] == 'cancelled' else
        'success_after_retention' if retained and finish['status']['success'] and finish['failure'] is None else
        'physical_failure' if finish['failure'] is not None or any(r['terminated'] for r in episode['rows']) else
        'budget_truncated')
    require(saved['end_reason'] == cycle['result']['end_reason'] == teacher_reason,
            'C34 real teacher termination/retention differs')
    transitions = saved['transitions']
    for transition, event in zip(transitions, events):
        interval = [transition['control_start'], transition['control_end']]
        require(transition['schema'] == 'd1-c34-actual-event-transition-v1'
                and transition['source_transition30_schema'] == 'd1-c30-event-transition-v1'
                and transition['control_range'] == interval
                and transition['global_control_range'] == [offset+x for x in interval]
                and transition['skill31_latch'] == event['skill31_latch'],
                'C34 actual event latch/range differs')
    macros = verify_macro_receipts34(episode, events, saved, direction,
               binding['actuator_ids'], LIMITS, teacher_reason,
               distance_m=distance_m)
    require(len(transitions) == saved['event_transition_count'] == len(events)
            == cycle['result']['actual_event_transitions'],
            'C34 actual-D event count differs')
    overlap = [r['control_index'] for r in side_rows
        if np.linalg.norm(r['skill30']['foot_reference_velocity_mps'][:2]) > 1e-10
        and abs(r['skill30']['foot_reference_velocity_mps'][2]) > 1e-10]
    requests = [r['control_index'] for r in episode['rows'] if r['cancel_requested30']]
    cancel_valid = (bool(overlap and requests == [overlap[0]+1]
        and cycle['result']['cancel_control_index'] == requests[0]
        and all(e['control_index'] <= requests[0] for e in events)) if is_cancel else not requests)
    gates = task_report['gates']
    mandatory = dict(entry=gates['entry'], side_physical_safety=gates['side_physical_safety'],
        rolling_physical_safety=gates['rolling_physical_safety'],
        no_termination_or_truncation=not any(r['terminated'] or r['truncated'] for r in episode['rows']),
        zero_raw_and_servo_motion=gates['all_raw_servo_motion_zero'],
        one_side_within_1500=gates['side_controls_within_1500'],
        safe_landed_handoff=gates['safe_landed_handoff'],
        exact_400_retention=gates['exact_400_retention_complete'],
        tail_stopped_loaded=gates['original_tail_stopped_and_loaded'],
        B22_predict_restored=gates['B22_predict_restored_in_retention'])
    safe = bool(all(mandatory.values()) and reference_report['horizontal_native_geometry_passed']
                and cancel_valid)
    success = bool(safe and not is_cancel and task_report['base_task_gates_passed'])
    cancel_ok = bool(safe and is_cancel and task_report['base_task_gates_passed'])
    kind = ('safe_cancel' if cancel_ok else 'success' if success else
            'controlled_failure' if safe and not is_cancel else 'fatal_incomplete_or_unsafe')
    online = saved_document(folder/'task31.json')
    first_side = side_rows[0]
    q0 = np.asarray(episode['states']['qpos'][first_side['control_index']], dtype=float)
    yaw0 = float(first_side['info']['controller_record']['diagnostic_before']['start_yaw_rad'])
    actual_delta = direction*distance_m*np.array([-np.sin(yaw0), np.cos(yaw0), 0.])
    actual_target = q0[:2]+actual_delta[:2]
    require(online == cycle['online_task'] == cycle['result']['online_task']
            and online['schema'] == 'd1-c34-online-task-v1'
            and online['inherited_physical_classifier'] == 'd1-c31-online-task-v1'
            and online['target_distance_m'] == distance_m
            and close(online['actual_start_delta_m'], actual_delta)
            and close(online['target_xy_m'], actual_target)
            and online['mandatory_gates'] == mandatory
            and online['safety_complete'] is all(mandatory.values())
            and online['online_success'] is success
            and online['online_safe_cancel'] is cancel_ok
            and online['online_controlled_failure_eligible'] is (kind == 'controlled_failure')
            and cycle['terminal_kind'] == cycle['result']['terminal_kind'] == kind,
            'C34 online task differs from independently classified physical case')
    for key in ('signed_lateral_m', 'final_xy_error_m', 'final_yaw_error_rad',
                'finish_support_margin_m', 'side_duration_s', 'retention_ratio',
                'max_rollback_m', 'cancel_to_handoff_controls'):
        require(online[key] is None if task_report[key] is None
                else close(online[key], task_report[key]),
                'C34 online task scalar differs: '+key)
    learning_cycle = copy.deepcopy(cycle['learning_cycle'])
    require(learning_cycle is not None
            and learning_cycle['cycle_index'] == cycle_index
            and learning_cycle['terminal_kind'] == kind
            and close(learning_cycle['actual_seconds'], episode['controls']*.01)
            and learning_cycle['macros'] == transitions
            and all(learning_cycle[key] is True for key in
                    ('archive_valid', 'native_valid', 'safe_handoff', 'retention_complete')),
            'C34 learner ledger differs from actual closed case')
    learning_cycle['independent_learning_eligible'] = safe and kind in (
        'success', 'controlled_failure')
    report = dict(cycle_index=cycle_index, case_id=case_id, mode=mode,
        direction=direction, requested_initial_yaw_rad=cycle['requested_initial_yaw_rad'],
        record_valid=True, local_view=episode['local_view_proof31'],
        controls=episode['controls'], normal_native=5*episode['controls'],
        side_controls=episode['side_controls'], policy_predictions=episode['policy_predictions'],
        task=task_report, reference=reference_report, latches=latch_report,
        macros=macros, teacher_terminal_reason=teacher_reason,
        independent_terminal_kind=kind,
        independent_learning_eligible=learning_cycle['independent_learning_eligible'],
        mandatory_native_and_record_qualification=safe,
        qualification_passed=success or cancel_ok,
        online_mandatory_gates=mandatory)
    if history['last_index'] != episode['side_controls']:
        raise ValueError('C34 force proof did not cover every side control')
    report['beta34'] = float(beta34)
    report['distance_m'] = float(distance_m)
    report['inherited_local_physical_verifier'] = (
        'episode_read31.local_episode31 -> records27.check_episode on all controls/native')
    report['replaced_global_lookups'] = [
        'C27 recompute_side_torque -> force_read34.recompute_side_torque34',
        'C31 verify_latches31 -> latches_read33.verify_latches33 alpha=1',
        'C31 verify_reference30 -> reference_read33.verify_reference33 plus actual-D foot',
        'C31 score_skill30 -> score_read34.score_skill34 actual-D goal/progress',
        'C31 verify_macro_receipts30 -> reward_read34.verify_macro_receipts34 actual-D']
    return report, cycle, episode
