"""C38-aware extension of the frozen read35.ledger_read35.check_ledger35.

The frozen checker hardcodes the C35 contract's own case count into two exact-equality
checks: `session['model_limits']['predict'] == 3001` and `predictions+32 <= 3032`. Those are
`5*(prepare_controls+retention_controls)+1` and `+32` for C35's 5-case budget. Every other
number in the function is already a loose `<=` bound against the (unchanged, inherited)
global API caps, so it holds for C38's smaller 2-case campaign without modification.

This generalises the ceiling from the SAME session fields the rest of the frozen function
already trusts (`prepare_controls`, `retention_controls`, `len(session['cases'])`), rather
than hardcoding a new literal, so it reduces to the exact frozen formula when case count is 5
and C35's 3001/3032 are recovered exactly (a pure test checks this).

Everything else is the frozen `check_ledger35` body, copied verbatim.
"""
from __future__ import annotations

from pathlib import Path

from ledger_read35 import STATIC_KEYS35, model_count_valid35, static_bounds35
from reader26 import require


def predict_ceiling38(session: dict) -> tuple[int, int]:
    """(model_limits.predict ceiling, predict+probe-row ceiling) for this case count."""
    per_case = session['prepare_controls']+session['retention_controls']
    cases = len(session['cases'])
    return cases*per_case+1, cases*per_case+32


def check_ledger38(worker: dict, episodes: list[dict], session: dict,
                   construction: dict) -> dict:
    count = len(episodes)
    result = worker['result']
    cases_limit = len(session['cases'])
    require(1 <= count <= cases_limit and count == result['attempted_cases']
            and count == len(result['cases'])
            and result['closed_cases'] == sum(
                e['cycle']['result']['failure'] is None for e in episodes)
            and all(result['cases'][i] == e['cycle']['result']
                    for i, e in enumerate(episodes))
            and all(episode['cycle']['case_index'] == i for i, episode in enumerate(episodes)),
            'C38 ledger omitted/reordered an attempted case')
    partial_cases = [i for i, e in enumerate(episodes) if e['fatal_partial']]
    require(partial_cases in ([], [count-1])
            and (not partial_cases or worker['failure'] is not None),
            'C38 fatal partial did not stop on the last attempted case')
    complete = worker['failure'] is None and not partial_cases
    controls = sum(e['closed_prefix_controls'] for e in episodes)
    native = sum(len(e['native']) for e in episodes)
    predictions = sum(e['cycle']['result']['policy_predictions'] for e in episodes)
    returned_side = sum(e['side_controls'] for e in episodes)
    predict_cap, predict_rows_cap = predict_ceiling38(session)
    require(controls == result['controls'] <= session['control_limit'] == 9000
            and native == result['normal_native'] <= session['normal_native_cap'] == 45000
            and 5*controls <= native <= 5*(controls+int(bool(partial_cases)))
            and predictions == result['policy_predictions'] <= predict_cap-1
            and returned_side <= 6000
            and sum(e['cycle']['result']['pair_exchanges'] for e in episodes)
                == result['pair_exchanges'],
            'C38 actual global prefix/control/native/model case sums differ')

    access = worker['side_access']
    require(access['schema'] == 'd1-c35-side-static-access-v1'
            and access['open_case'] is None and access['prepares'] == 0
            and access['rejected'] == 0 and access['within_actual_bounds'] is True
            and len(access['closed_cases']) == count
            and access['limits'] == session['static_API_global_caps']
            and access['actual_T_P_bounds'] == static_bounds35(
                access['computes'], access['starts']),
            'C38 side static owner/count ceiling differs')
    scopes = access['scopes']
    expected_begin = expected_starts = expected_computes = 0
    summed_static = {key: 0 for key in STATIC_KEYS35}
    case_rows = []
    for i, episode in enumerate(episodes):
        cycle = episode['cycle']
        receipt = cycle['side_access']
        require(receipt == access['closed_cases'][i]
                and receipt['case_index'] == i
                and receipt['passed'] is True,
                'C38 case static closure differs from global owner')
        before, actual = receipt['before'], receipt['actual']
        t, s = actual['computes'], actual['starts']
        begin, end = before['scope_begin'], receipt['scope_end']
        require(begin == expected_begin and end == begin+t+s
                and s in (0, 1) and 0 <= t <= 1200
                and before['starts'] == expected_starts
                and before['computes'] == expected_computes
                and before['counts'] == summed_static
                and receipt['actual_static_bounds'] == static_bounds35(t, s),
                'C38 case scope partition or T/S reservation differs')
        completed_side = episode['side_controls']
        require(completed_side <= t <= completed_side+int(episode['fatal_partial'])
                and (s == 1 if completed_side else True),
                'C38 side static compute count differs from returned side controls')
        selected = scopes[begin:end]
        require(len(selected) == s+t and [r['kind'] for r in selected]
                == ['start']*s+['compute']*t,
                'C38 start/compute scope order differs')
        completed_scopes = selected[s:s+completed_side]
        require([r['info']['controller_record']['access_scope_receipt']
                 for r in episode['side_rows']] == completed_scopes,
                'C38 returned side controls detached from counted compute scopes')
        for j, scope in enumerate(selected):
            is_start = j < s
            bound = static_bounds35(int(not is_start), int(is_start))
            require(scope['physical_arrays_unchanged'] is True
                    and scope['native_boundary_unchanged'] is True
                    and scope['static_ccd_returned'] is True
                    and scope['static_ccd_delta']['ccd_attempts']
                        == scope['static_ccd_delta']['ccd_returns'] >= 0
                    and scope['start_count'] == expected_starts+s
                    and scope['compute_count'] == expected_computes+max(0, j-s+1)
                    and all(0 <= scope['calls'][k] <= bound[k] for k in STATIC_KEYS35)
                    and (scope['failure'] is None or
                         (episode['fatal_partial'] and j == len(selected)-1)),
                    'C38 static query scope/physical nonmutation differs')
        local = {k: sum(scope['calls'][k] for scope in selected)
                 for k in STATIC_KEYS35}
        require(all(local[k] == receipt['static_counts'][k]
                    <= receipt['actual_static_bounds'][k] for k in STATIC_KEYS35),
                'C38 per-case actual static API work differs')
        for key in STATIC_KEYS35:
            summed_static[key] += local[key]
        expected_begin, end_actual = end, receipt['scope_end']
        expected_starts += s
        expected_computes += t
        case_rows.append(dict(case_index=i, returned_controls=episode['closed_prefix_controls'],
            returned_native=len(episode['native']), returned_side_controls=completed_side,
            static_computes=t, starts=s, static_counts=local,
            scope_range=[begin, end_actual], fatal_partial=episode['fatal_partial']))
    require(expected_begin == len(scopes) and expected_starts == access['starts']
            and expected_computes == access['computes']
            and access['counts'] == summed_static
            and all(summed_static[k] <= access['actual_T_P_bounds'][k]
                    <= access['limits'][k] for k in STATIC_KEYS35),
            'C38 cumulative static API scope/count closure differs')

    c, py, guard = worker['C_final'], worker['python'], worker['native_guard']
    require(c['construction_attempts'] == c['construction_returns'] == 2
            and c['construction_limit'] == 2 and c['control_limit'] == 45000
            and c['native_construction_caller_verified'] is True
            and c['phase'] == c['target_model'] == c['target_data'] == 0
            and c['violations'] == 0
            and c['ccd_attempts'] == c['ccd_returns']
            and (c['ccd_attempts'] == 0 or c['native_ccd_caller_verified'] is True)
            and c['control_returns'] == native
            and native <= c['control_attempts'] <= native+int(not complete),
            'C38 compiler/native C counters or caller phase differ')
    caller = c['first_control_step_caller_dladdr']
    functions = caller['path']
    require(caller['offset'] == 777003
            and caller['address'] == c['first_control_step_caller']
            and caller['address']-caller['base'] == caller['offset']
            and Path(functions).name == '_functions.cpython-310-x86_64-linux-gnu.so'
            and session['source_hashes'][functions]['sha256']
                == construction['proof']['elf_hashes'][functions]
            and any(slot['elf'] == functions and slot['symbol'] == 'mj_step'
                    and slot['passed'] is True
                    for slot in construction['proof']['jump_slots']),
            'C38 actual normal-step caller differs from sealed E return site')
    require(py['control_completed'] == controls
            and controls <= py['control_attempted'] <= controls+int(not complete)
            and py['native_returned'] == native
            and native <= py['native_attempted'] <= native+int(not complete)
            and c['control_attempts'] <= py['native_attempted']
            and py['native_attempted'] <= c['control_attempts']+int(not complete)
            and py['clock_advanced_substeps'] >= native
            and py['clock_advanced_substeps'] <= py['native_attempted']
            and py['forbidden_entries'] == 0
            and (py['fatal_latched'] is False if complete else
                 type(py['fatal_latched']) is bool)
            and (py['control_attempted'] == controls or py['fatal_latched'] is True)
            and guard['native_returned'] == native
            and native <= guard['native_attempted'] <= native+int(not complete)
            and native-int(not complete) <= guard['native_checked'] <= native
            and (guard['failure'] is None or not complete)
            and len(py['segments']) == len(guard['segments']) == count,
            'C38 Python/native guard cumulative attempts or partial suffix differ')
    native_offset = control_offset = 0
    for i, episode in enumerate(episodes):
        cycle, segment = episode['cycle'], episode['segment']
        pseg, gseg = py['segments'][i], guard['segments'][i]
        n, nn = episode['closed_prefix_controls'], len(episode['native'])
        require(gseg == segment['native_segment']
                and segment['global_control_begin'] == control_offset
                and segment['global_control_end'] == control_offset+n
                and segment['global_native_begin'] == native_offset
                and segment['global_native_end'] == native_offset+nn
                and pseg['name'] == gseg['name'] == 'c35_'+session['cases'][i]['id']
                and pseg['limit'] == gseg['control_limit'] == 1800
                and pseg['completed'] == n
                and n <= pseg['attempted'] <= n+int(episode['fatal_partial'])
                and pseg['native_returned'] == gseg['native_returned'] == nn
                and nn <= pseg['native_attempted'] <= nn+int(episode['fatal_partial'])
                and nn <= gseg['native_attempted'] <= nn+int(episode['fatal_partial'])
                and cycle['native_range'] == [native_offset, native_offset+nn]
                and cycle['control_range'] == [control_offset, control_offset+n],
                'C38 per-case Python/guard/archive segment accounting differs')
        control_offset += n
        native_offset += nn

    models = worker['model_calls']
    counts = models['counts']
    expected = dict(load=1, torch_load=3, predict=predictions+1,
                    forward=0, evaluate_actions=0, predict_values=0,
                    backward=0, learn=0, train=0, save=0)
    require(models['limits'] == session['model_limits']
            and set(counts) <= set(expected)
            and all(model_count_valid35(counts, k, expected[k], complete) for k in expected)
            and counts['predict']['rows_returned'] == predictions+32
            and predictions+32 <= counts['predict']['rows_attempted']
                <= predictions+32+int(not complete)
            and models['actor_rows'] == predictions+32
            and models['critic_rows'] == 0
            and counts['load']['phases']['probe']['returned'] == 1
            and counts['torch_load']['phases']['probe']['returned'] == 3
            and counts['predict']['phases']['probe']['returned'] == 1
            and counts['predict']['phases']['probe']['rows'] == 32
            and counts['predict']['phases']['control']['returned'] == predictions
            and counts['predict']['phases']['control']['rows'] == predictions,
            'C38 B22-only model calls, 32-row probe or no-learning ledger differs')
    require(expected['predict'] <= session['model_limits']['predict'] == predict_cap
            and predictions+32 <= predict_rows_cap
            and all(session['model_limits'][k] == expected[k]
                    for k in ('load', 'torch_load', 'forward', 'evaluate_actions',
                              'predict_values', 'backward', 'learn', 'train', 'save')),
            'C38 frozen B22 model ceilings differ')

    addresses, edges = construction['data_addresses'], worker['copy_edges']
    require(set(addresses) == {'live', 'measurement', 'side_scratch'}
            and all(type(v) is int and v > 0 for v in addresses.values())
            and len(set(addresses.values())) == 3
            and addresses['side_scratch'] == access['scratch_address']
            and len({e['reset']['model_address'] for e in episodes}) == 1
            and all(e['reset']['data_address'] == addresses['live'] for e in episodes)
            and set(edges) == {'live_to_measurement', 'measurement_to_side_scratch', 'rejected'}
            and edges['rejected'] == 0
            and controls+count <= edges['live_to_measurement']
                <= controls+count+int(not complete)
            and edges['measurement_to_side_scratch'] == access['counts']['copy'],
            'C38 persistent model/three MjData identities or copy edges differ')
    return dict(passed=True, attempted_cases=count, closed_prefix_controls=controls,
        actual_normal_native=native, compiler_native=2,
        pending_control_attempts=py['control_attempted']-controls,
        pending_native_attempts=py['native_attempted']-native,
        native_checked=guard['native_checked'], B22_predict_calls=expected['predict'],
        B22_actor_rows=predictions+32, B22_critic_rows=0,
        actual_side_starts=access['starts'], actual_side_computes=access['computes'],
        actual_static_API=access['counts'], actual_data_objects=3,
        copy_edges=edges, original_counters_not_reset=True,
        failure_recorded=not complete, per_case=case_rows,
        static_queries_are_not_native_integrations=True)
