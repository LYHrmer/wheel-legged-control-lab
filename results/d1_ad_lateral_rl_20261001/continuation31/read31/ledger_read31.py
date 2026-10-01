"""C31 actual per-reset, native, scratch and learning call reconciliation."""
from __future__ import annotations

import math
from pathlib import Path
from reader26 import require, saved_document


def case_ledger31(run, session, worker, episode, report, index, offset):
    n, t = episode['controls'], episode['side_controls']
    p = report['macros']['actual_macro_transitions']
    receipt = episode['cycle_receipt31']['side_access']
    access = worker['side_access']
    require(access['closed_cycles'][index] == receipt and receipt['actual'] == dict(starts=1, computes=t, prepares=p),
            'actual side case ledger differs from consumed scopes/latches')
    before, after = receipt['before'], receipt['after']
    bounds = dict(copy=t+4*p, forward=112*t+236*p, jacBody=96*t+192*p,
                  jac=4*t, fullM=t, objectVelocity=2*t+1)
    require(receipt['actual_static_bounds'] == bounds and receipt['passed'] is True,
            'C31 actual static bounds differ')
    begin, end = receipt['scope_range']
    require(begin == before['scopes'] and end-begin == t+1,
            'C31 static scope range is not actual start plus compute')
    scopes = access['scopes'][begin:end]
    require(len(scopes) == t+1 and scopes[0]['kind'] == 'start'
            and [s['kind'] for s in scopes[1:]] == ['compute']*t
            and scopes[1:] == episode['original_side_scopes31'], 'actual side scope references differ')
    prior_p = before['prepares']
    for j, scope in enumerate(scopes):
        require(scope['start_count'] == before['starts']+1
                and scope['compute_count'] == before['computes']+j
                and prior_p <= scope['prepare_count'] <= before['prepares']+p
                and scope['physical_arrays_unchanged'] is True and scope['native_boundary_unchanged'] is True
                and scope['static_ccd_returned'] is True and scope['failure'] is None
                and scope['static_ccd_delta']['ccd_attempts'] == scope['static_ccd_delta']['ccd_returns'] >= 0
                and scope['calls']['objectVelocity'] == (1 if j == 0 else 2),
                'C31 live-state/native fence or actual scope counter differs')
        prior_p = scope['prepare_count']
    require(prior_p == before['prepares']+p
            and all(after[k] == before[k]+v for k, v in receipt['actual'].items()),
            'C31 side cumulative scope counters reset')
    for key, limit in bounds.items():
        actual = sum(s['calls'][key] for s in scopes)
        require(actual == receipt['static_counts'][key] == after['counts'][key]-before['counts'][key]
                and 0 <= actual <= limit, 'C31 actual static API work differs: '+key)
    budget = saved_document(Path(run)/f'budget_cycle_{index:02d}.json')
    require(budget == worker['result']['budget']['closed'][index]
            and budget['cycle_index'] == index and budget['global_control_begin'] == offset
            and budget['global_native_begin'] == 5*offset
            and budget['global_control_end'] == offset+n and budget['global_native_end'] == 5*(offset+n)
            and budget['actual_controls'] == n and budget['actual_normal_native'] == 5*n
            and budget['actual_macros'] == p and budget['complete'] is True
            and budget['control_limit'] == min(2200, session['control_limit']-offset),
            'actual cumulative budget closure differs from saved physical work')
    return dict(controls=n, normal_native=5*n, side_controls=t, macros=p,
        policy_predictions=episode['policy_predictions'], static_counts=receipt['static_counts'],
        model_address=episode['original_reset31']['model_address'],
        data_address=episode['original_reset31']['data_address'], scope_range=[begin, end])


def close_ledgers31(run, session, worker, construction, cases, learning_counts, sampled_macros):
    result = worker['result']
    train = session['arm'] == 'train'
    count = 64 if train else 10
    require(len(cases) == result['cycles'] == count, 'successful run omitted preregistered actual cycles')
    n = sum(c['controls'] for c in cases)
    t = sum(c['side_controls'] for c in cases)
    p = sum(c['macros'] for c in cases)
    predictions = sum(c['policy_predictions'] for c in cases)
    require(n == result['controls'] <= session['control_limit']
            and 5*n == result['normal_native'] and p == result['true_macros'] <= session['macros_limit']
            and predictions == result['policy_predictions'] <= (38400 if train else 6000)
            and all(c['policy_predictions'] == 600 for c in cases)
            and len({c['model_address'] for c in cases}) == len({c['data_address'] for c in cases}) == 1,
            'actual global controls/macros/rolling predictions or single live model identity differs')
    c, py, native = worker['C_final'], worker['python'], worker['native_guard']
    caller = c['first_control_step_caller_dladdr']
    functions = caller['path']
    require(caller['offset'] == 777003 and caller['address'] == c['first_control_step_caller']
            and caller['address']-caller['base'] == caller['offset']
            and Path(functions).name == '_functions.cpython-310-x86_64-linux-gnu.so'
            and session['source_hashes'][functions]['sha256'] == construction['proof']['elf_hashes'][functions]
            and any(slot['elf'] == functions and slot['symbol'] == 'mj_step' and slot['passed'] is True
                    for slot in construction['proof']['jump_slots']), 'actual normal-step caller differs from closed E return site')
    require(c['construction_attempts'] == c['construction_returns'] == result['compiler_native'] == 2
            and c['control_attempts'] == c['control_returns'] == 5*n
            and c['phase'] == c['target_model'] == c['target_data'] == c['violations'] == 0
            and c['native_construction_caller_verified'] is True
            and c['ccd_attempts'] == c['ccd_returns']
            and (c['ccd_attempts'] == 0 or c['native_ccd_caller_verified'] is True)
            and py['control_attempted'] == py['control_completed'] == n
            and py['native_attempted'] == py['native_returned'] == py['clock_advanced_substeps'] == 5*n
            and py['forbidden_entries'] == 0 and py['fatal_latched'] is False
            and native['native_attempted'] == native['native_returned'] == native['native_checked'] == 5*n
            and native['failure'] is None, 'C31 global native/compiler/Python ledger differs')
    access = worker['side_access']
    bounds = dict(copy=t+4*p, forward=112*t+236*p, jacBody=96*t+192*p,
                  jac=4*t, fullM=t, objectVelocity=2*t+count)
    hard = {k: v*count for k, v in dict(copy=1516, forward=168944, jacBody=144768,
                                        jac=6000, fullM=1500, objectVelocity=3001).items()}
    require(access['starts'] == count and access['computes'] == t and access['prepares'] == p
            and access['rejected'] == 0 and access['within_actual_bounds'] is True
            and access['open_cycle'] is None and len(access['closed_cycles']) == count
            and access['actual_T_P_bounds'] == bounds and access['limits'] == hard
            and cases[0]['scope_range'][0] == 0 and cases[-1]['scope_range'][1] == len(access['scopes'])
            and all(a['scope_range'][1] == b['scope_range'][0] for a,b in zip(cases,cases[1:])),
            'C31 global static permission or scope partition differs')
    for key in bounds:
        require(access['counts'][key] == sum(row['static_counts'][key] for row in cases)
                <= bounds[key] <= hard[key], 'global static API sum differs: '+key)
    addresses, edges = construction['data_addresses'], worker['copy_edges']
    require(set(addresses) == {'live', 'measurement', 'side_scratch'} and len(set(addresses.values())) == 3
            and all(type(v) is int and v > 0 for v in addresses.values())
            and addresses['live'] == cases[0]['data_address'] and addresses['side_scratch'] == access['scratch_address']
            and set(edges) == {'live_to_measurement', 'measurement_to_side_scratch', 'rejected'}
            and edges['rejected'] == 0 and edges['live_to_measurement'] >= n
            and edges['measurement_to_side_scratch'] == access['counts']['copy'], 'C31 data ownership/copy fences differ')
    steps = learning_counts.get('actual_optimizer_steps', 0)
    attempts = learning_counts.get('actual_evaluated_minibatches', 0)
    counts = worker['model_calls']['counts']
    expected = dict(load=1, torch_load=4, predict=predictions+1, backward=steps,
        forward=0, evaluate_actions=0, predict_values=0, learn=0, train=0, save=0)
    require(worker['model_calls']['limits'] == session['model_limits'] and set(counts) <= set(expected)
            and all(counts.get(k, {}).get('attempted', 0) == counts.get(k, {}).get('returned', 0) == value
                    for k, value in expected.items())
            and counts['predict']['rows_attempted'] == counts['predict']['rows_returned'] == predictions+32
            and worker['model_calls']['actor_rows'] == predictions+32
            and worker['model_calls']['critic_rows'] == 0,
            'C31 actual B22 and global torch/backward calls differ')
    for phase, expected_loads in (('probe', 3), ('lateral_final' if train else 'lateral_load', 1)):
        require(counts['torch_load']['phases'][phase]['attempted']
                == counts['torch_load']['phases'][phase]['returned'] == expected_loads,
                'B22 versus lateral checkpoint loads not separated')
    probe = 32 if train else 16
    lateral_expected = dict(construct=2 if train else 1,
        actor_rows=sampled_macros+learning_counts.get('actual_pre_actor_rows', 0)
            +learning_counts.get('actual_post_kl_actor_rows', 0)+probe,
        value_rows=(sampled_macros if train else 0)+learning_counts.get('actual_pre_value_rows', 0)+probe,
        optimizer_step=steps, save=int(train))
    lateral = worker['lateral_model_calls']
    require(lateral == result['lateral_model_calls'] and lateral['actual_parameter_objects'] == (2 if train else 1)
            and set(lateral['counts']) <= set(lateral_expected)
            and all(lateral['counts'].get(k, {}).get('attempted', 0)
                == lateral['counts'].get(k, {}).get('returned', 0) == value
                <= lateral['limits'][k] for k, value in lateral_expected.items()),
            'C31 event policy/real update/probe model row ledger differs')
    phases_expected = {'control': dict(actor_rows=sampled_macros, value_rows=sampled_macros if train else 0),
        'lateral_update': dict(actor_rows=learning_counts.get('actual_pre_actor_rows', 0)+learning_counts.get('actual_post_kl_actor_rows', 0),
                              value_rows=learning_counts.get('actual_pre_value_rows', 0), optimizer_step=steps),
        'lateral_final' if train else 'lateral_load': dict(actor_rows=probe, value_rows=probe,
                              construct=1, save=int(train))}
    if train:
        phases_expected['construct'] = dict(construct=1)
    for phase, expected_phase in phases_expected.items():
        actual_phase = lateral['phases'].get(phase, {})
        require(set(actual_phase) <= set(expected_phase)
                and all(actual_phase.get(k, {}).get('attempted', 0) == actual_phase.get(k, {}).get('returned', 0) == value
                        for k, value in expected_phase.items()), 'C31 model calls in wrong phase: '+phase)
    require(set(lateral['phases']) <= set(phases_expected)
            and result['actual_optimizer_steps'] == steps <= 64
            and result['actual_minibatch_attempts'] == attempts <= 64
            and result['actual_sampler_actor_rows'] == sampled_macros
            and result['actual_sampler_value_rows'] == (sampled_macros if train else 0)
            and result['actual_rng_samples'] == (sampled_macros if train else 0), 'C31 actual attempts/sample ledger differs')
    reset = saved_document(Path(run)/'reset_yaw_receipt31.json')
    require(reset['calls'] == len(reset['receipts']) == count, 'extra/missing reset yaw calls')
    for i, row in enumerate(reset['receipts']):
        yaw = 0. if train else session['evaluation_cases'][i]['initial_yaw_rad']
        quaternion = None if yaw == 0 else [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
        require(row == dict(call_index=i, requested_initial_yaw_rad=yaw,
            forwarded_base_quaternion_wxyz=quaternion, original_reset_called_once=True,
            phase='plant_provider_controller_reset_before_prepare'), 'C31 reset yaw forwarding differs')
    return dict(passed=True, actual_cycles=count, controls=n, normal_native=5*n, compiler_native=2,
        true_macro_decisions=p, side_controls=t, rolling_B22_predictions=predictions,
        actual_optimizer_steps=steps, actual_minibatch_attempts=attempts,
        actual_lateral_model_calls=lateral_expected, actual_static_API=access['counts'],
        pure_native_shape_reconstructions=5*n, actual_data_objects=3,
        original_global_counters_not_reset=True, static_queries_are_not_integrations=True)
