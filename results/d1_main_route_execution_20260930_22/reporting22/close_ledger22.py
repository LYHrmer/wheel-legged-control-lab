"""Close actual C22 worker counters from saved receipts; stdlib only, no replay.

An absent/unreserved phase contributes zero. A reserved or started phase without
closed worker/host receipts is an error, never silently counted as zero. Scores
and scientific claims belong to the separately verified evaluation summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

C = Path(__file__).resolve().parent.parent
CONTRACT = 'C22_fixed_B_eval_repair_v1'
PHASES = {'train_B_1': ('per_training_worker_limits', 'training'),
          'development_1': ('development_worker_limits', 'eval'),
          'final_1': ('final_worker_limits', 'eval')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing regular evidence: '+str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            size += len(block)
            digest.update(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def close_ledger():
    inputs = {}
    def read(path):
        path = Path(path)
        value = json.loads(path.read_text())
        inputs[str(path)] = identity(path)
        return value
    def call(calls, name):
        return calls.get(name, {}).get('returned', 0)

    spec = read(C/'spec22.json')
    require(spec['execution_contract_id'] == CONTRACT, 'wrong C22 contract')
    limits = spec['whole_path_limits']
    require((limits['controls'], limits['normal_native'], limits['compiler_native'],
             limits['training_controls'], limits['cold_workers']) == (107968,539840,6,32768,3),
            'fixed C22 budget changed')
    totals = dict.fromkeys(('physical_controls','normal_native','compiler_native','actor_rows',
        'critic_rows','optimizer_steps','ppo_load','torch_load','worker_processes',
        'reserved_controls','independent_reader_nominal_intervals_verified'), 0)
    api_totals = {}
    rows = {}
    require({path.parent.name for path in C.glob('*/session.json')} <= set(PHASES),
            'unplanned C22 worker session exists')
    for name,(limit_key,kind) in PHASES.items():
        folder, reservation_path = C/name, C/(name+'_reservation.json')
        if not folder.exists() and not reservation_path.exists():
            rows[name] = {'status':'not_executed_not_reserved','physical_controls':0,
                          'normal_native':0,'compiler_native':0,'reserved_controls':0}
            continue
        require(folder.is_dir() and not folder.is_symlink() and reservation_path.is_file(),
                'partial/unclosed reservation: '+name)
        session = read(folder/'session.json')
        host = read(folder/'host_receipt.json')
        worker = read(folder/'worker_receipt.json')
        reservation = read(reservation_path)
        plan_path = Path(session['plan_path'])
        plan = read(plan_path)
        bound = spec[limit_key]
        require(session['execution_contract_id'] == worker['execution_contract_id'] == CONTRACT
                and plan['execution_contract_id'] == CONTRACT and plan['status'] == 'GO'
                and session['stage'] == 1 and session['reservation_key'] == name
                and session['mode'] == ('train' if kind == 'training' else 'eval')
                and session['control_limit'] == reservation['controls'] == bound['controls']
                and session['normal_native_limit'] == reservation['normal_native'] == bound['normal_native']
                and session['compiler_native_limit'] == reservation['compiler_native'] == 2
                and reservation['reservation_not_refundable'] is True
                and reservation['retry_permitted'] is False
                and session['plan_identity'] == reservation['plan_identity'] == identity(plan_path)
                and session['source_hashes'] == {**plan['inputs'],str(plan_path):identity(plan_path)}
                and plan['arms'][name]['controls'] == bound['controls']
                and worker['session_identity'] == identity(folder/'session.json'),
                'worker reservation/source identity differs: '+name)
        require(host['reservation_key'] == name and host['arm'] == session['arm']
                and host['postcheck_complete'] is True and host['no_live_owned_processes'] is True
                and host['fully_reserved_budget_closed'] is True and host['retry_permitted'] is False
                and host['changed_sources'] == [] and host['cleanup_errors'] == []
                and host['elapsed_s'] <= bound['hard_s'], 'worker host did not close: '+name)
        py, native, c = worker['python'], worker['native_guard'], worker['C_final']
        count = py['control_completed']
        require(type(count) is int and 0 <= count <= bound['controls']
                and py['control_attempted'] == count
                and py['native_attempted'] == py['native_returned'] == 5*count
                and py['native_failed'] == py['forbidden_entries'] == 0
                and c['control_attempts'] == c['control_returns'] == 5*count
                and c['construction_attempts'] == c['construction_returns'] == 2
                and c['violations'] == c['phase'] == c['target_model'] == c['target_data'] == 0
                and native['native_attempted'] == native['native_returned']
                    == native['native_checked'] == 5*count,
                'actual C/Python/native ledger mismatch: '+name)
        calls = worker['model_calls']['counts']
        for api,stats in calls.items():
            require(stats['attempted'] == stats['returned']
                    and stats['rows_attempted'] == stats['rows_returned']
                    and stats['returned'] <= worker['model_calls']['limits'][api],
                    'unfinished or over-budget model API: '+name+'/'+api)
            total = api_totals.setdefault(api, {'calls':0,'rows':0})
            total['calls'] += stats['returned']
            total['rows'] += stats['rows_returned']
        row = {'status':'closed','physical_controls':count,'normal_native':5*count,
            'compiler_native':2,'actor_rows':worker['model_calls']['actor_rows'],
            'critic_rows':worker['model_calls']['critic_rows'],'optimizer_steps':0,
            'ppo_load':call(calls,'load'),'torch_load':call(calls,'torch_load'),
            'worker_processes':1,'reserved_controls':bound['controls'],
            'model_api_counts':calls,'worker_execution_complete':worker['execution_complete'],
            'worker_failure':worker['failure'],'host_failure':host['failure'],
            'host_elapsed_s':host['elapsed_s'],'independent_reader_valid':False,
            'independent_reader_nominal_intervals_verified':0}
        successful = (worker['execution_complete'] is True and worker['failure'] is None
            and worker['cleanup_errors'] == [] and worker['warnings'] == []
            and worker['archive_failed'] is False and native['failure'] is None
            and host['failure'] is None and host['exit_code'] == 0)
        reader_path = C/(name+'_readback.json')
        reader_host_path = C/(name+'_readback_host_receipt.json')
        if successful:
            result, reader_host = read(reader_path), read(reader_host_path)
            reader_source = C/('read_training22.py' if kind == 'training' else 'read_eval22.py')
            inputs[str(reader_source)] = identity(reader_source)
            require(reader_host['exit_code'] == 0 and reader_host['failure'] is None
                    and reader_host['output_identity'] == identity(reader_path)
                    and reader_host['reader_identity'] == identity(reader_source)
                    == session['source_hashes'][str(reader_source)]
                    and reader_host['controls'] == reader_host['model_calls'] == 0
                    and reader_host['elapsed_s'] <= bound['reader_hard_s']
                    and result['execution_contract_id'] == CONTRACT
                    and result['run'] == str(folder) and result['source_closure_verified'] is True,
                    'independent readback/host/source closure differs: '+name)
            row['independent_reader_valid'] = True
            row['reader_elapsed_s'] = reader_host['elapsed_s']
            row['independent_reader_nominal_intervals_verified'] = result['pure_nominal_servo_intervals_verified']
            if kind == 'training':
                require(result['complete_controls'] == count == 32768
                        and result['training_valid'] is True and result['learning_valid'] is True
                        and result['archive_valid'] is True
                        and result['reader_model_calls'] == result['reader_physics_calls'] == 0,
                        'training readback count/validity differs')
                clip = read(folder/'clip_audit.json')
                learning = read(folder/'training/learning_receipt.json')
                blocks = read(folder/'training/training_blocks_manifest.json')
                manifest = read(folder/'final_checkpoint_manifest.json')
                actual = learning['actual']
                require(actual == result['actual_learning']['actual'] == worker['result']['training']['actual']
                        and clip['optimizer_attempted'] == clip['optimizer_returned']
                        == actual['optimizer_steps'] == call(calls,'backward')
                        == manifest['actual_optimizer_steps']
                        and clip['native_clip_attempted'] == clip['native_clip_returned'] == 2*actual['optimizer_steps']
                        and len(clip['evaluated_batches']) == actual['evaluated_minibatches']
                        == call(calls,'evaluate_actions')
                        and count == sum(x['rows'] for x in blocks['numeric_blocks'])
                        == sum(x['rows'] for x in blocks['full_control_blocks20']) == blocks['gaussian_records']
                        and blocks['pending_gaussian_control_index'] is None
                        and call(calls,'forward') == count and call(calls,'predict') == 3
                        and row['actor_rows'] == count+256*actual['evaluated_minibatches']+96
                        and row['critic_rows'] == count+256*actual['evaluated_minibatches']+call(calls,'predict_values'),
                        'training independent numeric/learning/model counters disagree')
                row.update(optimizer_steps=actual['optimizer_steps'],actual_learning=actual,
                    coverage_valid=result['coverage_valid'],coverage=result['coverage'],
                    checkpoint_model_identity=manifest['files']['final_model.zip'],
                    geometry_resets_verified=result['geometry_resets_verified'])
            else:
                require(result['actual_completed_controls'] == count
                        and result['actual_normal_native_returns'] == 5*count
                        and result['actual_compiler_native_returns'] == 2
                        and result['physical_calls_performed_by_reader'] == 0
                        and result['model_loaded_by_reader'] is False,
                        'evaluation independent physical ledger differs: '+name)
                cases, floors = worker['result']['cases'], worker['result']['floors']
                regular = sum(x['policy_predictions'] for x in cases)+sum(x['completed_controls'] for x in floors.values())
                require(count == sum(x['completed_controls'] for x in cases)+sum(x['completed_controls'] for x in floors.values())
                        and call(calls,'predict') == regular+3
                        and row['actor_rows'] == regular+96 and row['critic_rows'] == 0
                        and all(call(calls,k) == 0 for k in ('learn','train','forward','evaluate_actions','backward','predict_values','save')),
                        'evaluation case/floor/probe model counts disagree: '+name)
                row.update(case_count=len(cases),floor_count=len(floors),
                    regular_policy_prediction_rows=regular,
                    full_comparison_executed=worker['result']['full_comparison_executed'])
        else:
            # Retain failed consumption; no success or zero-consumption inference.
            row['status'] = 'closed_failure_independent_qualification_unavailable'
            if kind == 'training' and (folder/'clip_audit.json').is_file():
                clip = read(folder/'clip_audit.json')
                row['optimizer_steps'] = clip['optimizer_returned']
        require(row['ppo_load'] <= bound['ppo_load'] and row['torch_load'] <= bound['torch_load']
                and row['actor_rows'] <= bound['actor_rows_max']
                and row['critic_rows'] <= bound.get('critic_rows_max',bound.get('critic_rows',0)),
                'model row budget exceeded: '+name)
        for key in totals:
            totals[key] += row[key]
        rows[name] = row
    for key,cap in (('physical_controls','controls'),('normal_native','normal_native'),
                    ('compiler_native','compiler_native'),('actor_rows','actor_rows'),
                    ('critic_rows','critic_rows'),('optimizer_steps','optimizer_max'),
                    ('ppo_load','ppo_load'),('torch_load','torch_load'),('worker_processes','cold_workers')):
        require(totals[key] <= limits[cap], 'C22 total budget exceeded: '+key)
    require(totals['normal_native'] == 5*totals['physical_controls'], 'aggregate 5T closure differs')
    reuse, geometry = read(C/'a_reuse_22.json'), read(C/'finite_geometry_22.json')
    require(reuse['passed'] is True and reuse['A_new_training_controls_in_C22'] == 0
            and reuse['model_calls'] == reuse['physics_calls'] == 0
            and geometry['passed'] is True and geometry['model_calls'] == geometry['physics_calls'] == 0
            and geometry['new_pure_servo_advance_count'] == 6400
            and geometry['old_C21_actual_pure_servo_calls'] == 184800,
            'historical reuse or separate pure geometry accounting differs')
    optional = {}
    for name in ('first1024_pair_22.json','final_execution_decision_22.json'):
        path = C/name
        if path.exists():
            value = read(path)
            optional[name] = {key:value[key] for key in ('schema','passed','final_execution_allowed') if key in value}
    historical_C20 = read(C.parent/'continuation20/execution_ledger_20.json')
    for path,expected in inputs.items():
        require(identity(path) == expected, 'evidence changed while closing ledger: '+path)
    return {'schema':'d1-c22-closed-execution-ledger-v1','execution_contract_id':CONTRACT,
        'workers':rows,'actual_totals':totals,'model_api_totals':api_totals,
        'contract_upper_bounds':limits,'actual_native_including_compiler':totals['normal_native']+totals['compiler_native'],
        'all_three_workers_executed':all(row['status'] != 'not_executed_not_reserved' for row in rows.values()),
        'all_executed_workers_independently_verified':all(row.get('independent_reader_valid',False)
            for row in rows.values() if row['status'] != 'not_executed_not_reserved'),
        'unused_reserved_controls_are_not_refunded':True,'unexecuted_phases_not_counted':True,
        'new_A_training_controls':0,'stage2_controls':0,'extra_trials':0,
        'historical_C20_all_workers_excluded_from_new_totals':historical_C20['actual_totals'],
        'historical_A20_is_a_subset_of_C20_not_an_additional_cost':True,
        'historical_A20_excluded_from_new_totals':{'training_controls':reuse['A_actual_training_controls'],
            'optimizer_steps':reuse['A_actual_optimizer_steps'],'checkpoint_sha256':reuse['A_checkpoint_sha256'],
            'source_reader':spec['A_reuse']['reader']},
        'historical_shared_parent_excluded_from_new_totals':{'ancestral_training_controls':spec['parent']['ancestral_training_controls'],
            'checkpoint_sha256':spec['parent']['model_sha256']},
        'pure_geometry_separate_from_robot_budget':{'historical_C21_servo_intervals':184800,
            'historical_C21_preflight_passed':False,'historical_C21_robot_controls':0,
            'historical_C21_model_calls':0,'historical_C21_compiler_native':0,
            'new_C22_preflight_servo_intervals':6400,'C21_plus_C22_preflight_servo_intervals':191200,
            'reused_qualified_intervals_not_reexecuted':geometry['reused_nominal_intervals'],
            'new_independent_reader_servo_intervals':totals['independent_reader_nominal_intervals_verified'],
            'reader_intervals_are_separate_from_preflight_6400_and_from_robot_controls':True},
        'optional_decision_evidence':optional,'inputs':inputs,
        'new_model_calls_by_this_script':0,'new_physics_calls_by_this_script':0,
        'new_servo_calls_by_this_script':0,'scientific_benefit_claim_not_inferred_from_ledger':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=C/'execution_ledger_22.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'ledger output already exists; use a new named snapshot')
    result = close_ledger()
    result['ledger_source_identity'] = identity(Path(__file__))
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'output':str(args.output),'identity':identity(args.output),
                      'actual_totals':result['actual_totals']},sort_keys=True))


if __name__ == '__main__':
    main()
