"""Saved-only C38 provenance/counts plus unchanged controller/native/task mathematics."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location('numpy38_contract',Path(__file__).with_name('run_b22_numpy38.py'))
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)
c = r.c


def verify_bootstrap(value,session,expected_session_identity):
    worker = r.load_file('numpy38_cold_proof_reader',r.ROOT/'scripts/run_b22_numpy_worker38.py')
    c.require(value.get('schema') == 'b22-numpy38-bootstrap-v1' and value.get('status') == 'COMPLETE'
        and value.get('session_identity') == expected_session_identity
        and value.get('execution_contract_id') == r.CONTRACT and value.get('qualification_granted') is False
        and value.get('forbidden_modules_absent') is True
        and value.get('forbidden_prefixes') == sorted(r.backend.FORBIDDEN)
        and type(value.get('environment_constructions')) is int and value['environment_constructions'] == 0
        and type(value.get('numpy_actor_bundle_loads')) is int and value['numpy_actor_bundle_loads'] == 0,
        'C38 cold bootstrap identity/zero-model environment differs')
    for key,keys in (('actual_counts',('controls','normal_native','compiler_native')),
                     ('actual_nn_counts',('predict_calls','predict_rows','ppo_load','torch_load'))):
        c.require(isinstance(value.get(key),dict) and set(value[key]) == set(keys)
            and all(type(v) is int and v == 0 for v in value[key].values()),'C38 bootstrap nonzero/unknown counters')
    c.require(value.get('nominal_cache') == {'hits':0,'misses':0,'maxsize':1,'currsize':0},
              'C38 cold bootstrap nominal cache was used')
    c.require(all(type(v) is int for v in value['nominal_cache'].values()),'C38 nominal-cache counters need exact integers')
    for key in ('state_before','state_after'):
        worker.zero_state38(value[key])
    proof,sources = value['binding_proof'],session['source_hashes']
    worker.zero_state38(proof['counters'])
    c.require(proof.get('passed') is True and proof.get('dso_path') == session['library']
        and proof.get('dso_sha256') == sources[session['library']]['sha256']
        and len(proof.get('jump_slots',())) == 4
        and len({(row['elf'],row['symbol'],row['relocation_offset']) for row in proof['jump_slots']}) == 4
        and sorted(row['symbol'] for row in proof['jump_slots']) == ['mj_step','mj_step','mj_step','mjc_ccd']
        and all(row.get('passed') is True and row.get('target') == row.get('expected_wrapper')
            and row.get('elf_sha256') == sources[row['elf']]['sha256'] for row in proof['jump_slots'])
        and all(sources[name]['sha256'] == digest for name,digest in proof['elf_hashes'].items()),
        'C38 saved ELF/GOT/source binding differs')
    origins = value.get('module_origins')
    c.require(isinstance(origins,dict) and origins and all(path in sources for path in origins.values())
        and all(name.partition('.')[0] in ('numpy','mujoco') for name in origins),
        'C38 bootstrap actual dependency origins differ')
    return True


def verify_numpy_ledger(result,session,count):
    calls = result.get('model_calls',{})
    p = calls.get('counts',{}).get('predict',{})
    expected = {'attempted':count+1,'returned':count+1,'rows_attempted':count+32,'rows_returned':count+32}
    c.require(calls.get('schema') == 'b22-numpy38-model-api-ledger-v1' and calls.get('backend') == 'NumPy'
        and calls.get('limits') == session.get('model_limits') == r.backend.API_LIMITS
        and all(type(v) is int for values in (calls['limits'],session['model_limits']) for v in values.values())
        and calls.get('forbidden_modules_absent') is True
        and all(type(calls.get(key)) is int and calls[key] == expected_value for key,expected_value in (
            ('actor_rows',count+32),('critic_rows',0),('numpy_actor_bundle_loads',1),('ppo_load_calls',0),('torch_load_calls',0)))
        and set(calls.get('counts',{})) == {'predict'}
        and all(type(p.get(key)) is int and p[key] == value for key,value in expected.items())
        and json.dumps(p.get('phases'),sort_keys=True,allow_nan=False) == json.dumps(
            {'probe':{'attempted':1,'returned':1,'rows':32},
             'control':{'attempted':count,'returned':count,'rows':count}},sort_keys=True),
        'C38 actual NumPy/probe/control/forbidden-framework ledger differs')


def native_ledger_checker(base):
    """Change only old load=1/torch_load=3 expectations; retain all native/owner checks."""
    path = r.ROOT/'runtime/d1_b22/research/continuation26/root26/reader26.py'
    node = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body
                              if isinstance(n,ast.FunctionDef) and n.name == 'check_ledgers'))
    changed = []
    for item in ast.walk(node):
        if isinstance(item,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'expected' for t in item.targets):
            for keyword in item.value.keywords:
                if keyword.arg in ('load','torch_load'):
                    keyword.value = ast.Constant(0)
                    changed.append(keyword.arg)
    c.require(sorted(changed) == ['load','torch_load'],'C38 old ledger seam changed')
    namespace = dict(base.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)  # noqa: S102 -- SHA-bound pure ledger with two explicit expectation seams
    return namespace['check_ledgers']


def verify_origins(origins,sources):
    paths = origins.get('module_origins')
    c.require(origins.get('schema') == 'b22-numpy38-runtime-origins-v1'
        and origins.get('forbidden_modules_absent') is True
        and origins.get('forbidden_prefixes') == sorted(r.backend.FORBIDDEN)
        and origins.get('interpreter_backend') == 'NumPy'
        and type(origins.get('ppo_load_calls')) is int and origins['ppo_load_calls'] == 0
        and type(origins.get('torch_load_calls')) is int and origins['torch_load_calls'] == 0
        and isinstance(paths,dict) and paths and all(path in sources for path in paths.values())
        and all(name.partition('.')[0] in ('numpy','gymnasium','mujoco') for name in paths),
        'C38 actual no-framework dependency origins differ')


def metadata_checker(prior):
    path = r.ROOT/'scripts/read_b22_suite31.py'
    text = ast.get_source_segment(path.read_text(),next(n for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name == 'script_sources'))
    substitutions = {
        "'C31_B22_fixed_suite_v1'":repr(r.CONTRACT),
        "'scripts/run_b22_worker31.py'":"'scripts/run_b22_numpy_worker38.py'",
        "c.require(all(path in sources for path in origins['module_origins'].values())\n              and all(row['identity_verified'] is True and row['defining_source'] in sources\n                      for row in origins['synthetic_module_aliases'].values()), 'actual module origin left closure')":
            'verify_origins(origins,sources)',
        "expected_origins['engine_binding'] = session['binding_module']":
            "expected_origins['engine_binding'] = session['binding_module']\n    expected_origins.pop('rl16_learning_11')\n    expected_origins['b22_numpy_backend38'] = str(c.ROOT/'scripts/b22_numpy_backend38.py')"}
    for old,new in substitutions.items():
        c.require(text.count(old) == 1,'C38 explicit metadata seam changed: '+old)
        text = text.replace(old,new)
    namespace = dict(prior.__dict__,verify_origins=verify_origins)
    namespace['c'] = SimpleNamespace(**{**vars(c),'case_completion':r.case_completion,
        'model_limits':lambda actor,controls,keys:dict(r.backend.API_LIMITS)})
    exec(compile(text,str(path),'exec'),namespace)  # noqa: S102 -- pinned pure metadata function; explicit new backend identity
    return namespace['script_sources']


def verify_load_proof(proof,session,release):
    helper = r.backend.helper36()
    expected = r.backend.export_identities38()
    c.require(proof.get('schema') == 'b22-numpy38-load-proof-v1'
        and proof.get('execution_contract_id') == r.CONTRACT and proof.get('backend') == 'NumPy'
        and proof.get('model_sha256') == session['checkpoint_sha256'] == helper.MODEL_SHA
        and proof.get('metadata_sha256') == helper.METADATA_SHA
        and proof.get('source_policy_state_sha256') == helper.POLICY_STATE_SHA
        and proof.get('export36_artifacts') == release['export36'] == expected
        and proof.get('probe_rows') == 32 and type(proof['probe_rows']) is int
        and proof.get('probe_prediction_calls') == 1 and type(proof['probe_prediction_calls']) is int
        and proof.get('probe_actions_within_tolerance') is True and proof.get('absolute_tolerance') == 1e-6
        and proof.get('verified_without_engine_or_reset') is True
        and proof.get('control_qualification_granted') is False
        and type(proof.get('ppo_load_calls')) is int and proof['ppo_load_calls'] == 0
        and type(proof.get('torch_load_calls')) is int and proof['torch_load_calls'] == 0,
        'C38 NumPy source/bundle/original32 load proof differs')
    for key in ('clipped16_max_absolute_error','half_gain_max_absolute_error'):
        value = proof.get(key)
        c.require(type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1e-6,
                  'C38 original32 numerical tolerance exceeded')


def check_gain_row(row):
    import numpy as np
    network = np.asarray(row.get('network_output16'),dtype=np.float32)
    action = np.asarray(row.get('policy_input_action'))
    c.require(type(row.get('gain38')) is float and row['gain38'] == .5
        and network.shape == action.shape == (16,) and np.isfinite(network).all() and np.isfinite(action).all()
        and (abs(network) <= 1.).all() and np.array_equal(action,(network*np.float32(.5)).astype(np.float64))
        and row.get('action16') == row.get('raw_action16') == row['policy_input_action']
        and row.get('policy_predict_called') is True,'C38 saved NumPy output/actual half-gain action differs')


def read_stage(release_path,name):
    release = r.validate_release(release_path)
    stage = next((row for row in r.stages() if row['name'] == name),None)
    c.require(stage is not None,'C38 unknown fixed profile')
    run = Path(release['plan']['output'])/name
    prior = r.old35.pinned('numpy38_original_math31',r.ROOT/'scripts/read_b22_suite31.py')
    base,step,scorer = prior.helpers(release)
    base.check_ledgers = native_ledger_checker(base)
    session,worker,construction,count = metadata_checker(prior)(run,stage,release,base)
    c.require(type(session.get('gain38')) is float and session['gain38'] == worker['result'].get('gain38') == .5,
              'C38 session/actual gain differs')
    owner = r.load_file('numpy38_derivation_reader',r.ROOT/'scripts/run_b22_numpy_worker38.py')
    c.require(c.read_json(run/'derived_sources38.json') == owner.derivation_proof38(owner.derived_source38()),
              'C38 executed numeric-owner derivation differs')
    verify_bootstrap(c.read_json(run/'bootstrap38_report.json'),session,c.identity(run/'session.json'))
    verify_numpy_ledger(worker['result'],session,count)
    verify_load_proof(base.saved_document(run/'strict_load.json'),session,release)
    rows = 0
    for row in base.saved_rows(run/'episode_0/controls.jsonl.gz'):
        c.require(row['control_index'] == rows,'C38 saved control ordering differs')
        check_gain_row(row)
        rows += 1
    c.require(rows == count,'C38 complete gain/action archive missing')
    canonical,pairing = prior.canonical_script(run,stage,session,worker,construction,count,base,step)
    score = scorer.score_case(canonical,expected_seed=stage['seed'],mirror=False)
    files = prior.artifact_identities(run,base)
    for filename in ('bootstrap38_report.json','derived_sources38.json'):
        files[filename] = c.identity(run/filename)
    return {'schema':'b22-numpy38-stage-readback-v1','status':'COMPLETE','execution_contract_id':r.CONTRACT,
        'stage':name,'release_identity':c.identity(release_path),'reader_identity':c.identity(Path(__file__)),
        'actual_counts':{'controls':count,'normal_native':5*count,'compiler_native':2},
        'numpy_predict_calls':count+1,'numpy_predict_rows':count+32,'ppo_load':0,'torch_load':0,
        'gain_ledger_rows':rows,'cases':[score],'pairing_identity':pairing,'artifacts':files,
        'reader_physics_calls':0,'reader_nn_forward_calls':0,'qualification_granted':False}


def compare_reports(reports,references):
    pairs = []
    for stage,after in zip(r.stages(),reports):
        before = references[stage['name']]
        c.require(after['stage'] == before['stage'] == stage['name']
            and after['pairing_identity'] == before['pairing_identity'],'C38 historical same seed/reset/robot pairing differs')
        a,b = before['cases'][0],after['cases'][0]
        c.require(a['windows'] == b['windows'],'C38 complete scoring windows differ')
        for score in (a,b):
            c.require(score['case_id'] == stage['case_id'] and score['seed'] == stage['seed'] and score['mirror'] is False
                and score['actor'] == 'final_policy' and score['windows']['control_cap'] == 1600
                and score['windows']['hold'] == stage['hold'] and score['windows']['drive'] == [175,stage['release_tick']],
                'C38 same command/hold/drive horizon differs')
        full = all(score['record']['full_horizon'] is True and score['completed_controls'] == 1600
                   and score['rl_terms']['drive_complete'] is True for score in (a,b))
        values = {}
        for key in ('sse_total','torque_cost_sum'):
            old,new = a['rl_terms'][key],b['rl_terms'][key]
            c.require(all(type(value) in (int,float) and math.isfinite(value) and value >= 0 for value in (old,new)),
                      'C38 invalid SSE/normalized torque cost')
            values[key] = {'old35':old,'numpy38':new,'compatibility_limit':old*1.001,
                'within_0p1_percent':full and new <= old*1.001,'ratio':new/old if old else None}
        pairs.append({'stage':stage['name'],'full_horizon_pair':full,'old35_task_passed':a['task_passed'],
            'numpy38_task_passed':b['task_passed'],'metrics':values,
            'backend_compatibility_passed':full and b['task_passed'] is True
                and all(row['within_0p1_percent'] for row in values.values()),'old35_case':a,'numpy38_case':b})
    c.require(len(reports) == len(pairs) == 2,'C38 two fixed profiles required')
    return {'pairs':pairs,'all_two_full_horizons':all(row['full_horizon_pair'] for row in pairs),
        'all_two_tasks_passed':all(row['numpy38_task_passed'] is True for row in pairs),
        'backend_compatibility_passed':all(row['backend_compatibility_passed'] for row in pairs),
        'relative_metric_upper_factor':1.001,'qualification_granted':False,
        'scope':'two fixed NumPy backend feedback cases; no bitwise trajectory, full-matrix, RL contribution or real-robot qualification claim'}


def compare(release_path):
    release = r.validate_release(release_path)
    output = Path(release['plan']['output'])
    refs = r.reference35()['reports']
    reports,identities = [],{}
    for stage in r.stages():
        path = output/(stage['name']+'_readback.json')
        saved = c.read_json(path)
        r.stage_counts(stage,saved,c.identity(release_path),c.identity(Path(__file__)))
        for name,expected in saved['artifacts'].items():
            c.require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and c.identity(output/stage['name']/name) == expected,'C38 saved raw artifact changed')
        reports.append(saved)
        identities[path.name] = c.identity(path)
    return {'schema':'b22-numpy38-comparison-v1','status':'COMPLETE','release_identity':c.identity(release_path),
        'stage_reports':identities,'new_actual_counts':r.counted({row['stage']:row['actual_counts'] for row in reports}),
        'old35_outer_actual_counts':None,'historical_total_resources_known':False,
        **compare_reports(reports,refs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--stage',choices=tuple(row['name'] for row in r.stages()))
    modes.add_argument('--compare',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args(argv)
    value = compare(args.release) if args.compare else read_stage(args.release,args.stage)
    c.write_json(args.report,value)
    print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
