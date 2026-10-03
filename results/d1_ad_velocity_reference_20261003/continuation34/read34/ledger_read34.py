"""C34 cumulative model/native/static-API ledger; no budget refund."""
from pathlib import Path

from reader26 import require, saved_document


def global_ledger34(run, session, worker, construction, audits):
    result = worker['result']
    count = len(audits)
    n = sum(row['controls'] for row in audits)
    t = sum(row['side_controls'] for row in audits)
    p = sum(row['macros'] for row in audits)
    predictions = sum(row['policy_predictions'] for row in audits)
    require(1 <= count <= 10 and count == result['closed_cases']
            and n == result['controls'] <= 22000
            and 5*n == result['normal_native'] <= 110000
            and p == result['true_macros'] <= 40
            and predictions == result['policy_predictions'] == 600*count <= 6000
            and result['budget']['limits'] == dict(cycles=10, controls=22000, macros=40)
            and len(result['budget']['closed']) == count
            and result['budget']['macros'] == p
            and len({row['model_address'] for row in audits}) == 1
            and len({row['data_address'] for row in audits}) == 1,
            'C34 actual case/control/native/model budget differs')
    c, py, native = worker['C_final'], worker['python'], worker['native_guard']
    require(c['construction_attempts'] == c['construction_returns'] == 2
            and c['control_attempts'] == c['control_returns'] == 5*n
            and c['phase'] == c['target_model'] == c['target_data'] == c['violations'] == 0
            and c['native_construction_caller_verified'] is True
            and c['ccd_attempts'] == c['ccd_returns']
            and (c['ccd_attempts'] == 0 or c['native_ccd_caller_verified'] is True)
            and py['control_attempted'] == py['control_completed'] == n
            and py['native_attempted'] == py['native_returned'] == py['clock_advanced_substeps'] == 5*n
            and py['forbidden_entries'] == 0 and py['fatal_latched'] is False
            and native['native_attempted'] == native['native_returned'] == native['native_checked'] == 5*n
            and native['failure'] is None,
            'C34 C/Python/native/compiler ledger differs')
    caller = c['first_control_step_caller_dladdr']
    lib = caller['path']
    require(caller['address'] == c['first_control_step_caller']
            and caller['offset'] == 777003
            and caller['address']-caller['base'] == caller['offset']
            and session['source_hashes'][lib]['sha256'] == construction['proof']['elf_hashes'][lib]
            and any(row['elf'] == lib and row['symbol'] == 'mj_step' and row['passed'] is True
                    for row in construction['proof']['jump_slots']),
            'C34 normal native step caller differs')
    access = worker['side_access']
    bounds = dict(copy=t+4*p, forward=112*t+236*p,
                  jacBody=96*t+192*p, jac=4*t, fullM=t,
                  objectVelocity=2*t+count)
    hard = dict(copy=15160, forward=1689440, fullM=15000,
                jac=60000, jacBody=1447680, objectVelocity=30010)
    require(access['starts'] == count and access['computes'] == t
            and access['prepares'] == p and access['rejected'] == 0
            and access['within_actual_bounds'] is True and access['open_cycle'] is None
            and len(access['closed_cycles']) == count
            and access['actual_T_P_bounds'] == bounds and access['limits'] == hard
            and audits[0]['scope_range'][0] == 0
            and audits[-1]['scope_range'][1] == len(access['scopes'])
            and all(a['scope_range'][1] == b['scope_range'][0]
                    for a, b in zip(audits, audits[1:])),
            'C34 actual static API scope/ceiling differs')
    for key in bounds:
        require(access['counts'][key] == sum(row['static_counts'][key] for row in audits)
                <= bounds[key] <= hard[key],
                'C34 actual static API count differs: '+key)
    addresses, edges = construction['data_addresses'], worker['copy_edges']
    require(set(addresses) == {'live', 'measurement', 'side_scratch'}
            and len(set(addresses.values())) == 3
            and addresses['live'] == audits[0]['data_address']
            and addresses['side_scratch'] == access['scratch_address']
            and set(edges) == {'live_to_measurement', 'measurement_to_side_scratch', 'rejected'}
            and edges['rejected'] == 0 and edges['live_to_measurement'] >= n
            and edges['measurement_to_side_scratch'] == access['counts']['copy'],
            'C34 one-plant/three-data copy fence differs')
    calls = worker['model_calls']
    counts = calls['counts']
    expected = dict(load=1, torch_load=3, predict=predictions+1,
        forward=0, evaluate_actions=0, predict_values=0, backward=0,
        learn=0, train=0, save=0)
    require(calls['limits'] == session['model_limits']
            and set(counts) <= set(expected)
            and all(counts.get(key, {}).get('attempted', 0)
                    == counts.get(key, {}).get('returned', 0) == value
                    for key, value in expected.items())
            and counts['predict']['rows_attempted'] == counts['predict']['rows_returned'] == predictions+32
            and calls['actor_rows'] == predictions+32 and calls['critic_rows'] == 0
            and worker.get('lateral_model_calls') is None,
            'C34 B22-only model/no-training ledger differs')
    reset = saved_document(Path(run)/'reset_yaw_receipt31.json')
    require(reset['calls'] == len(reset['receipts']) == count,
            'C34 extra or missing reset calls')
    for i, row in enumerate(reset['receipts']):
        require(row == dict(call_index=i, requested_initial_yaw_rad=0.,
            forwarded_base_quaternion_wxyz=None, original_reset_called_once=True,
            phase='plant_provider_controller_reset_before_prepare'),
            'C34 reset/yaw forwarding differs')
    return dict(passed=True, cases=count, controls=n, normal_native=5*n,
                compiler_native=2, macros=p, side_controls=t,
                B22_control_predictions=predictions, B22_probe_rows=32,
                new_lateral_actor_calls=0, new_lateral_value_calls=0,
                new_optimizer_steps=0, data_objects=3,
                actual_static_API=access['counts'])
