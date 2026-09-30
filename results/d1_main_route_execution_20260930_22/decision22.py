"""Fixed final execution prerequisites and descriptive comparisons; never loads policies or physics."""
import argparse
import json
from pathlib import Path
from worker22 import identity, write

C = Path(__file__).resolve().parent


def under(ratio, bound):
    if ratio['denominator'] > 0:
        return ratio['numerator'] / ratio['denominator'] <= bound
    return ratio['numerator'] == 0 and bound >= 1.


def decide(a, b, dev, paired):
    pool = dev['equal_case_pool']
    regression = dev['original_six_pool']
    gates = {
        'two_valid_training_records': a['training_valid'] and b['training_valid'],
        'effective_coverage_both': a['coverage_valid'] and b['coverage_valid'],
        'first1024_pair': paired['passed'],
        'dev_source_and_old_reuse': dev['source_closure_verified'] and dev['regression_reuse']['complete'],
        'all_development_records_complete_safe': pool['complete'],
        'regression_pool_complete_safe': regression['complete'],
    }
    if pool['complete']:
        cmp = pool['comparisons']
        gates.update(B_all_dev_tasks=pool['actors']['B']['all_tasks_passed'],
            dev_B_A_error=under(cmp['B/A']['error'], .90),
            dev_B_old_error=under(cmp['B/old']['error'], .95),
            dev_B_zero_cost=under(cmp['B/zero']['cost'], 1.20),
            dev_B_A_cost=under(cmp['B/A']['cost'], 1.10))
        for family in ('yaw', 'bumps'):
            for metric in ('drive_error_mean','hold_error_mean'):
                gates[family+'_'+metric] = under(pool['family_B_over_A'][family][metric], 1.02)
    else:
        gates['dev_numeric_gates'] = False
    if regression['complete']:
        gates['regression_B_zero_cost'] = under(regression['comparisons']['B/zero']['cost'], 1.20)
        gates['regression_B_A_cost'] = under(regression['comparisons']['B/A']['cost'], 1.10)
    original_ids = {'flat_0p6','flat_1p6','flat_1p2_yaw','bumps_0p4','rough_0p35',
                    'ramp_0p45_complete','flat_1p2_yaw_mirror'}
    rows = [r for r in dev['numeric_scores'] if r['experiment_actor']=='B' and r['case_id'] in original_ids]
    gates['B_all_original_seven'] = len(rows)==7 and {r['case_id'] for r in rows}==original_ids and all(r['task_passed'] for r in rows)
    passed = all(gates.values())
    comparable = all(gates[k] for k in ('two_valid_training_records','effective_coverage_both',
        'first1024_pair','dev_source_and_old_reuse','all_development_records_complete_safe','regression_pool_complete_safe'))
    execution_gates = {
        'valid_training_records': a['training_valid'] and b['training_valid'],
        'A_reuse_bridge_valid': dev['A_reuse']['passed'],
        'fresh_first1024_pair': paired['passed'],
        'source_closure_and_old_reuse': dev['source_closure_verified'] and dev['regression_reuse']['complete'],
        'both_floors_passed': set(dev['enabled_arms']) == {'A','B'}
            and set(dev['floors']) == {'A','B'}
            and all(row['numeric_gate_passed'] for row in dev['floors'].values()),
        'all_prelisted_development_records_present': len(dev['heldout_summaries']) == 30,
    }
    return {'schema':'d1-c22-fixed-final-execution-decision-v1',
            'execution_contract_id':'C22_fixed_B_eval_repair_v1',
            'execution_gates':execution_gates,
            'final_execution_allowed':all(execution_gates.values()),
            'development_descriptive_gates':gates,
            'development_descriptive_gates_passed':passed,
            'development_comparison_complete':comparable,
            'continue_training':False,'candidate_stage':1,
            'coverage_or_performance_selects_final':False,
            'final_scores_inspected':False,'new_model_or_physics_calls':0}


def verified_inputs():
    paths = [C.parent/'continuation20/train_A_1_readback.json', C/'train_B_1_readback.json', C/'development_1_readback.json']
    values, identities = [], {}
    for path in paths:
        hostpath = path.with_name(path.stem+'_host_receipt.json')
        host = json.loads(hostpath.read_text())
        if host['failure'] is not None or host['exit_code'] != 0 or host['output_identity'] != identity(path):
            raise ValueError('independent readback did not close: '+str(path))
        values.append(json.loads(path.read_text()))
        identities[str(path)] = identity(path)
        identities[str(hostpath)] = identity(hostpath)
    pairpath = C/'first1024_pair_22.json'
    paired = json.loads(pairpath.read_text())
    for path, expected in paired['inputs'].items():
        if identity(path) != expected:
            raise ValueError('paired warmup input changed')
    identities[str(pairpath)] = identity(pairpath)
    return values, paired, identities


def current_decision():
    values, paired, identities = verified_inputs()
    result = decide(*values, paired)
    result['inputs'] = identities
    return result


def validate_saved_decision(path):
    saved = json.loads(Path(path).read_text())
    if saved != current_decision():
        raise ValueError('decision differs from full frozen development and training gate evidence')
    return saved


def first_pair():
    import numpy as np
    inputs, fields = {}, {}
    initial_states, resets = [], []
    for arm in ('A','B'):
        folder=(C.parent/'continuation20/train_A_1/training' if arm == 'A' else C/'train_B_1/training')
        initial=folder/'training_episode_000000_initial_state.npz'
        reset=folder/'training_episode_000000_control_reset.json'
        for path in (initial,reset,initial.with_name(initial.name+'.manifest.json'),reset.with_name(reset.name+'.manifest.json')):
            inputs[str(path)]=identity(path)
        with np.load(initial,allow_pickle=False) as data:
            initial_states.append({key:data[key] for key in data.files})
        resets.append(json.loads(reset.read_text()))
    for key in ('qpos','qvel','ctrl','qacc_warmstart','observation'):
        x,y=initial_states[0][key],initial_states[1][key]
        fields['initial:'+key]=bool(x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes())
    fields['initial:full_control_reset']=resets[0]==resets[1]
    for category in ('numeric','gaussian'):
        folder = 'training_'+category+'_blocks'
        stem = 'controls' if category == 'numeric' else 'gaussian'
        paths = [base/folder/(stem+'_0000.npz') for base in
                 (C.parent/'continuation20/train_A_1/training', C/'train_B_1/training')]
        loaded = []
        for path in paths:
            inputs[str(path)] = identity(path)
            with np.load(path, allow_pickle=False) as data:
                loaded.append({k:data[k] for k in data.files})
        x,y = loaded
        if set(x) != set(y):
            raise ValueError('first1024 numeric schemas differ')
        for key in x:
            # Case and source recipe labels are stored outside these arrays.
            fields[category+':'+key] = bool(x[key].shape == y[key].shape and x[key].shape[0] == 1024
                and x[key].dtype == y[key].dtype and x[key].tobytes() == y[key].tobytes())
    result = {'schema':'d1-c22-first1024-byte-pair-v1','execution_contract_id':'C22_fixed_B_eval_repair_v1','inputs':inputs,'fields':fields,'passed':all(fields.values()),
              'model_calls':0,'physics_calls':0}
    write(C/'first1024_pair_22.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pair',action='store_true')
    args=parser.parse_args()
    if args.pair:
        result=first_pair()
    else:
        result=current_decision()
        write(C/'final_execution_decision_22.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
