"""Independent C31 episode proof using explicit, in-memory global/local views.

Original archives are hash-checked first and never rewritten. Only independently
verified counter offsets and the episode label are translated for frozen C27
numerical checks; physical values, actions, forces and clocks remain unchanged.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from types import FunctionType

import numpy as np
import records27
from reader26 import (require, close, saved_document, saved_arrays, saved_rows,
                      document, check_binding, geometry_map)
from reader27 import frozen_profile
from side_math27 import LIMITS
from reference_read30 import native_geometry30, verify_reference30
from reward_read30 import verify_macro_receipts30
from score_read30 import score_skill30
from latches_read31 import verify_latches31

KEYS = ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'observation', 'time')
COMPONENTS = dict(progress='progress_potential_difference', backtrack='backtrack_normalized',
    elapsed_s='elapsed_s', longitudinal_square_m2_s='longitudinal_square_integral_m2_s',
    longitudinal_square_normalized_s='longitudinal_normalized_square_integral_s',
    lateral_goal_square_m2_s='lateral_goal_error_square_integral_m2_s',
    lateral_goal_square_normalized_s='lateral_goal_error_normalized_square_integral_s',
    yaw_square_rad2_s='yaw_error_square_integral_rad2_s',
    yaw_square_normalized_s='yaw_error_normalized_square_integral_s',
    actual_torque_square_normalized_s='torque_normalized_square_integral_s')


def translate_boundary31(boundary, offset, expected_global_controls):
    require(type(offset) is int and 0 <= offset <= expected_global_controls
            and boundary['C_state']['control_attempts'] == boundary['C_state']['control_returns'] == 5*expected_global_controls
            and boundary['C_state']['construction_attempts'] == boundary['C_state']['construction_returns'] == 2
            and boundary['python']['control_attempted'] == boundary['python']['control_completed'] == expected_global_controls
            and boundary['python']['native_attempted'] == boundary['python']['native_returned'] == 5*expected_global_controls,
            'C31 global ledger was reset/refunded or contains missing controls')
    local = copy.deepcopy(boundary)
    for key in ('control_attempts', 'control_returns'):
        local['C_state'][key] -= 5*offset
    for key in ('control_attempted', 'control_completed'):
        local['python'][key] -= offset
    for key in ('native_attempted', 'native_returned'):
        local['python'][key] -= 5*offset
    return local


def construction31(run, session, repository):
    c = saved_document(Path(run)/'construction_receipt.json')
    proof = c['proof']
    require(proof['passed'] is True and proof['dso_path'] == session['library']
            and proof['dso_sha256'] == session['source_hashes'][session['library']]['sha256']
            and len(proof['jump_slots']) == 4 and all(r['passed'] is True for r in proof['jump_slots'])
            and c['nominal_cache']['misses'] == 1
            and c['C_state']['construction_attempts'] == c['C_state']['construction_returns'] == 2,
            'C31 cold two-compiler construction proof differs')
    binding = check_binding(c['geometry_binding'])
    geometry = geometry_map(c['compiled_geometry'], binding)
    reference = document(session['reference_construction_path'])
    require(c['geometry_binding'] == reference['geometry_binding']
            and c['compiled_geometry'] == reference['compiled_geometry'], 'C31 compiled robot/course differs')
    side = c['side_kinematic_binding']
    require({k: v for k, v in side.items() if k not in ('kinematics24', 'dynamics24', 'solver24')}
            == c['geometry_binding'], 'C31 compiled FK binding differs')
    kin = side['kinematics24']
    for key in ('geom_bodyid', 'geom_type', 'geom_size', 'geom_contype', 'geom_conaffinity', 'body_mass'):
        require(close(kin[key], c['geometry_binding'][key]), 'compiled FK array differs: '+key)
    require(close(np.asarray(side['dynamics24']['actuator_ctrlrange'])[binding['actuator_ids']],
                  np.stack((-LIMITS, LIMITS), axis=1))
            and c['side_profile'] == frozen_profile(Path(repository)/'scripts/d1_fast_side_step.py'),
            'C31 teacher profile or actual motor limits changed')
    return c, binding, geometry, kin


def local_episode31(run, session, construction, binding, geometry, kin, cycle_index, offset):
    folder = Path(run)/f'episode_{cycle_index}'
    reset = saved_document(folder/'reset.json')
    states = saved_arrays(folder/'states.npz')
    initial = saved_arrays(folder/'initial_state.npz')
    segment = saved_document(folder/'segment_receipt.json')
    cycle = saved_document(folder/'cycle_receipt.json')
    rows = list(saved_rows(folder/'controls.jsonl.gz'))
    n = len(rows)
    require(cycle['schema'] == 'd1-c31-cycle-receipt-v1' and cycle['cycle_index'] == cycle_index
            and segment['cycle_index'] == cycle_index and n == segment['completed_controls']
            and 0 < n <= 2200 and set(initial) == set(KEYS), 'C31 actual episode identity/count differs')
    for record in (segment, cycle['result']):
        require(record['global_control_begin'] == offset and record['global_control_end'] == offset+n
                and record['global_native_begin'] == 5*offset and record['global_native_end'] == 5*(offset+n),
                'C31 recorded global control/native offset differs')
    require(reset['seed'] == session['seed'] and cycle['archive_completed'] is True
            and cycle['independent_readback_pending'] is True, 'C31 incomplete episode archive')
    yaw = cycle['requested_initial_yaw_rad']
    require(yaw == reset['requested_initial_yaw_rad'] and yaw in (0., .04, -.04)
            and close(initial['qpos'][:3], session['spawn_position_m'], atol=1e-13)
            and close(initial['qpos'][3:7], [math.cos(yaw/2), 0., 0., math.sin(yaw/2)], atol=1e-13),
            'actual initial reset pose differs from declared finite yaw condition')
    require(type(reset['model_address']) is int and reset['model_address'] > 0
            and reset['data_address'] == construction['data_addresses']['live'], 'reset live model/data identity differs')
    translated_reset = copy.deepcopy(reset)
    translated_segment = copy.deepcopy(segment)
    translated_reset['before'] = translate_boundary31(reset['before'], offset, offset)
    translated_reset['after'] = translate_boundary31(reset['after'], offset, offset)
    translated_segment['boundary'] = translate_boundary31(segment['boundary'], offset, offset+n)
    translated_segment['episode_index'] = 0
    access = cycle['side_access']
    require(access['cycle_index'] == cycle_index and access['passed'] is True,
            'C31 side API case closure missing')
    before = access['before']
    raw_scopes = []
    for i, row in enumerate(rows):
        require(row['control_index'] == row['episode_tick'] == i and row['episode_index'] == cycle_index
                and row['global_control_index'] == offset+i
                and row['macro_events30'] == row['macro_events31'], 'C31 local/global row mapping differs')
        row['episode_index'] = 0
        if row['actor'] == 'traditional_side':
            adapter = row['info']['controller_record']
            require(adapter == row['control_stages18'], 'original controller aliases differ')
            scope = adapter['access_scope_receipt']
            raw_scopes.append(copy.deepcopy(scope))
            require(scope['compute_count'] == before['computes']+adapter['side_compute_index'],
                    'actual side API compute count does not match global/local mapping')
            scope['compute_count'] -= before['computes']
            scope['start_count'] -= before['starts']
            scope['prepare_count'] -= before['prepares']
            row['control_stages18'] = copy.deepcopy(adapter)
    native_by_name = {}
    native_index = 0
    for name in segment['native_segment']['native_files']:
        require(Path(name).name == name and name not in native_by_name, 'unsafe/repeated native block')
        block = list(saved_rows(folder/name))
        for item in block:
            proof = item['skill30_geometry']
            require(item['native_index'] == proof['native_index'] == proof['global_native_index31'] == 5*offset+native_index
                    and proof['cycle_index31'] == cycle_index
                    and proof['local_native_index31'] == native_index
                    and proof['local_control_index31'] == native_index//5
                    and proof['global_control_index31'] == offset+native_index//5,
                    'actual global native index/proof has a gap, duplicate or wrong cycle')
            item['native_index'] -= 5*offset
            proof['native_index'] -= 5*offset
            native_index += 1
        native_by_name[name] = block
    require(native_index == 5*n, 'C31 full original native count differs')
    # The frozen function receives no changed dynamics/state, only checked indices.
    docs = {'reset.json': translated_reset, 'segment_receipt.json': translated_segment}
    arrays = {'initial_state.npz': initial, 'states.npz': states}
    rowsets = {'controls.jsonl.gz': rows, **native_by_name}
    namespace = dict(records27.check_episode.__globals__)
    namespace.update(saved_document=lambda path: docs[Path(path).name],
                     saved_arrays=lambda path: arrays[Path(path).name],
                     saved_rows=lambda path: iter(rowsets[Path(path).name]))
    check = FunctionType(records27.check_episode.__code__, namespace, 'checked_local_episode31')
    episode = check(run, {**session, 'control_limit': 2200, 'side_arm': 'teacher'}, binding, geometry, kin)
    require(episode['controls'] == cycle['result']['completed_controls']
            and episode['policy_predictions'] == cycle['result']['policy_predictions']
            and cycle['result']['pure_fk_native_calls'] == 5*n, 'C31 actual per-case counted work differs')
    episode['original_side_scopes31'] = raw_scopes
    episode['original_reset31'] = reset
    episode['cycle_receipt31'] = cycle
    episode['original_segment31'] = segment
    episode['local_view_proof31'] = dict(cycle_index=cycle_index, global_control_offset=offset,
        global_native_offset=5*offset, side_compute_offset=before['computes'],
        original_archive_rewritten=False, physical_arrays_actions_forces_unchanged=True,
        translated_fields=['episode_index', 'C/Python integration counter offsets',
                           'native_index and skill30_geometry.native_index',
                           'side access cumulative start/compute/prepare counts'])
    return episode


def canonical_check(saved, computed, interval):
    expected = {key: computed[key] for key in COMPONENTS.values()}
    expected.update(control_range=interval, elapsed_controls=interval[1]-interval[0],
        reward_weights_defined=False, weighted_scalar_reward=None, torque_cost_is_energy=False,
        source_component_schema='d1-c30-macros-v1')
    require(set(saved) == set(expected), 'C31 canonical macro component schema differs')
    for key, value in expected.items():
        require(saved[key] == value if value is None or isinstance(value, (str, bool, list))
                else close(saved[key], value), 'C31 canonical component differs: '+key)


def physical_episode31(run, session, construction, binding, geometry, kin, original_spec,
                       cycle_index, offset, parameters):
    episode = local_episode31(run, session, construction, binding, geometry, kin, cycle_index, offset)
    cycle = episode['cycle_receipt31']
    folder = Path(run)/f'episode_{cycle_index}'
    saved = saved_document(folder/'macro_transitions31.json')
    events = saved['events']
    mode, direction = cycle['mode'], cycle['direction']
    require(saved['schema'] == 'd1-c31-actual-event-macros-v1'
            and saved['elapsed_controls'] == episode['controls']
            and saved['global_control_offset'] == offset and saved['direction'] == direction,
            'C31 saved macro episode identity differs')
    latches = verify_latches31(episode, events, construction, binding, cycle_index=cycle_index,
        direction=direction, mode=mode, rollout_parameters=parameters)
    minima, contacts = native_geometry30(episode, kin, binding)
    is_cancel = bool(cycle['result']['cancel_control_index'] is not None)
    case_id = ('cancel_' if is_cancel else 'zero_' if mode == 'zero' else 'skill_')+('left' if direction == 1 else 'right')
    reference = verify_reference30(episode, events, kin, binding, case_id, minima, contacts)
    task = score_skill30(episode, binding, geometry, kin, case_id, direction, original_spec)
    side_rows = [r for r in episode['rows'] if r['actor'] == 'traditional_side']
    finish = side_rows[-1]['info']['controller_record']['diagnostic_after']
    end = side_rows[-1]['control_index']+1
    retained = finish['done'] and episode['controls'] == end+400
    teacher_reason = ('safe_cancel' if retained and finish['failure'] == 'cancelled' else
        'success_after_retention' if retained and finish['status']['success'] and finish['failure'] is None else
        'physical_failure' if finish['failure'] is not None or any(r['terminated'] for r in episode['rows']) else
        'budget_truncated')
    require(saved['end_reason'] == cycle['result']['end_reason'] == teacher_reason,
            'C31 teacher termination/retention differs')
    original_transitions = copy.deepcopy(saved['transitions'])
    for transition, event in zip(original_transitions, events):
        interval = [transition['control_start'], transition['control_end']]
        require(transition['schema'] == 'd1-c31-actual-event-transition-v1'
                and transition['source_transition30_schema'] == 'd1-c30-event-transition-v1'
                and transition['control_range'] == interval
                and transition['global_control_range'] == [offset+x for x in interval]
                and transition['skill31_latch'] == event['skill31_latch'], 'C31 macro actual latch/range differs')
        transition['schema'] = transition['source_transition30_schema']
        transition['reward_components'] = transition['original_reward_components30']
    macros = verify_macro_receipts30(episode, events, original_transitions, direction,
                                    binding['actuator_ids'], LIMITS, teacher_reason)
    require(len(saved['transitions']) == saved['event_transition_count'] == len(events)
            == cycle['result']['actual_event_transitions'], 'C31 actual event count differs')
    for transition, computed in zip(saved['transitions'], macros['macro_components']):
        canonical_check(transition['reward_components'], computed, computed['control_range'])
    canonical_check(saved['full_case_components'], macros['full_scene_components'], [0, episode['controls']])
    require(set(saved['original_full_case_components30']) == set(COMPONENTS)
            and all(close(saved['original_full_case_components30'][a], macros['full_scene_components'][b])
                    for a, b in COMPONENTS.items()), 'C31 full-scene raw/native cost differs')
    overlap = [r['control_index'] for r in side_rows
        if np.linalg.norm(r['skill30']['foot_reference_velocity_mps'][:2]) > 1e-10
        and abs(r['skill30']['foot_reference_velocity_mps'][2]) > 1e-10]
    requests = [r['control_index'] for r in episode['rows'] if r['cancel_requested30']]
    cancel_valid = (bool(overlap and requests == [overlap[0]+1]
        and cycle['result']['cancel_control_index'] == requests[0]
        and all(e['control_index'] <= requests[0] for e in events)) if is_cancel else not requests)
    g = task['gates']
    mandatory = dict(entry=g['entry'], side_physical_safety=g['side_physical_safety'],
        rolling_physical_safety=g['rolling_physical_safety'],
        no_termination_or_truncation=not any(r['terminated'] or r['truncated'] for r in episode['rows']),
        zero_raw_and_servo_motion=g['all_raw_servo_motion_zero'],
        one_side_within_1500=g['side_controls_within_1500'], safe_landed_handoff=g['safe_landed_handoff'],
        exact_400_retention=g['exact_400_retention_complete'], tail_stopped_loaded=g['original_tail_stopped_and_loaded'],
        B22_predict_restored=g['B22_predict_restored_in_retention'])
    safe = bool(all(mandatory.values()) and reference['horizontal_native_geometry_passed'] and cancel_valid)
    success = bool(safe and not is_cancel and task['base_task_gates_passed'])
    cancel_ok = bool(safe and is_cancel and task['base_task_gates_passed'])
    kind = 'safe_cancel' if cancel_ok else 'success' if success else 'controlled_failure' if safe and not is_cancel else 'fatal_incomplete_or_unsafe'
    online = saved_document(folder/'task31.json')
    require(online == cycle['online_task'] == cycle['result']['online_task']
            and online['mandatory_gates'] == mandatory and online['safety_complete'] is all(mandatory.values())
            and online['online_success'] is success and online['online_safe_cancel'] is cancel_ok
            and online['online_controlled_failure_eligible'] is (kind == 'controlled_failure')
            and cycle['terminal_kind'] == cycle['result']['terminal_kind'] == kind,
            'online learning admission differs from independent physical/task classification')
    for key in ('signed_lateral_m', 'final_xy_error_m', 'final_yaw_error_rad', 'finish_support_margin_m',
                'side_duration_s', 'retention_ratio', 'max_rollback_m', 'cancel_to_handoff_controls'):
        require(online[key] is None if task[key] is None else close(online[key], task[key]),
                'online task scalar differs: '+key)
    learning_cycle = copy.deepcopy(cycle['learning_cycle'])
    require(learning_cycle is not None and learning_cycle['cycle_index'] == cycle_index
            and learning_cycle['terminal_kind'] == kind and close(learning_cycle['actual_seconds'], episode['controls']*.01)
            and learning_cycle['macros'] == saved['transitions']
            and all(learning_cycle[k] is True for k in ('archive_valid', 'native_valid', 'safe_handoff', 'retention_complete')),
            'learner cycle differs from actual closed episode')
    learning_cycle['independent_learning_eligible'] = safe and kind in ('success', 'controlled_failure')
    report = dict(cycle_index=cycle_index, case_id=cycle['case_id'], mode=mode, direction=direction,
        requested_initial_yaw_rad=cycle['requested_initial_yaw_rad'], record_valid=True,
        local_view=episode['local_view_proof31'], controls=episode['controls'], normal_native=5*episode['controls'],
        side_controls=episode['side_controls'], policy_predictions=episode['policy_predictions'],
        task=task, reference=reference, latches=latches, macros=macros,
        teacher_terminal_reason=teacher_reason, independent_terminal_kind=kind,
        independent_learning_eligible=learning_cycle['independent_learning_eligible'],
        mandatory_native_and_record_qualification=safe, qualification_passed=success or cancel_ok)
    return report, learning_cycle, episode
