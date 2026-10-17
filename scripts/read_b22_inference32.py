"""Saved-only, exact full-domain comparison for the independent B22 pair32."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import itertools
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location('inference32_contract', Path(__file__).with_name('run_b22_inference32.py'))
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)
STATE_KEYS = ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'observation', 'time')


def exact(first, second, label):
    """JSON values compare by exact numeric spelling; no absolute/relative tolerance."""
    encoded = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    c.require(encoded(first) == encoded(second), 'paired numerical value differs: '+label)


def byte_equal(first, second, label):
    import numpy as np
    a, b = np.asarray(first), np.asarray(second)
    c.require(a.dtype.kind in 'fiub' and b.dtype.kind in 'fiub'
              and bool(np.isfinite(a).all()) and bool(np.isfinite(b).all())
              and a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(),
              'paired array bytes/dtype/shape differ: '+label)


def check_inference(row, mode, gate):
    import numpy as np
    ledger = row['inference32']
    predicted = mode == 'baseline' or gate
    c.require(ledger.get('schema') == 'b22-gated-stateless-inference32-v1'
              and ledger.get('actor') == 'B' and ledger.get('checkpoint_sha256') == c.MODEL_SHA
              and ledger.get('action_gate_enabled') is gate
              and ledger.get('predict_called') is row.get('policy_predict_called') is predicted
              and ledger.get('policy_action_observed') is predicted, 'inference32 decision/prediction differs')
    action = np.asarray(row['action16'])
    c.require(action.shape == (16,) and action.dtype.kind in 'fiu'
              and np.isfinite(action).all() and (abs(action.astype(np.float64)) <= 1.).all(), 'invalid action16')
    exact(row['action16'], row['policy_input_action'], 'control policy input')
    exact(row['action16'], ledger['controller_input_action16'], 'ledger controller input')
    if predicted:
        exact(row['action16'], row['raw_action16'], 'observed SB3 policy action')
        exact(row['action16'], ledger['raw_policy_action16'], 'ledger observed policy action')
        c.require(ledger['reason'] == ('BASELINE_PREDICTED' if mode == 'baseline' else 'PREDICTED')
                  and ledger['policy_output_kind'] == 'SB3_clipped_deterministic_action', 'prediction provenance differs')
    else:
        c.require(not action.any() and row['raw_action16'] is None
                  and ledger['raw_policy_action16'] is None and ledger['reason'] == 'GATE_OFF'
                  and ledger['policy_output_kind'] == 'NOT_OBSERVED', 'skipped policy output was invented')
    return int(predicted)


def compare_numeric(baseline, gated, controls):
    """Stream every paired control/native row, while retaining only the NPZ states."""
    import numpy as np
    c.require(type(controls) is int and controls > 0, 'invalid comparison horizon')
    for key in STATE_KEYS:
        for data in (baseline, gated):
            a, initial = data['states'][key], data['initial'][key]
            c.require(a.shape == (controls+1, *initial.shape), 'missing full state endpoints')
            byte_equal(a[0], initial, 'initial state endpoint '+key)
        byte_equal(baseline['initial'][key], gated['initial'][key], 'paired initial '+key)
        byte_equal(baseline['states'][key], gated['states'][key], 'entire state horizon '+key)
    predictions = {'baseline': 0, 'gated': 0}
    off, native_count = 0, 0
    iterators = [iter(data['rows']()) for data in (baseline, gated)]
    natives = [iter(data['native']()) for data in (baseline, gated)]
    sentinel = object()
    for tick in range(controls):
        rows = [next(iterator, sentinel) for iterator in iterators]
        c.require(all(row is not sentinel for row in rows), 'control stream shorter than horizon')
        gate = rows[0]['control_stages18']['calculation']['action_enabled']
        c.require(type(gate) is bool and rows[1]['control_stages18']['calculation']['action_enabled'] is gate,
                  'paired original command gate differs')
        off += int(not gate)
        for mode, data, row in zip(('baseline','gated'), (baseline,gated), rows):
            c.require(row.get('control_index') == row.get('episode_tick') == tick and row.get('episode_index') == 0
                      and row.get('actor') == 'B' and row.get('checkpoint_sha256') == c.MODEL_SHA,
                      'control/checkpoint identity differs')
            predictions[mode] += check_inference(row, mode, gate)
            for key in STATE_KEYS:
                c.require(np.array_equal(np.asarray(row['pre_state'][key]), data['states'][key][tick])
                          and np.array_equal(np.asarray(row['post_state'][key]), data['states'][key][tick+1]),
                          'control/NPZ endpoint join differs: '+key)
            c.require(np.array_equal(np.asarray(row['input_observation99']), data['states']['observation'][tick]),
                      'actual inference input differs from saved observation99')
        for key in ('raw_command','input_observation99','control_stages18','native_actuator_traces',
                    'pre_state','post_state','reward','terminated','truncated'):
            exact(rows[0][key], rows[1][key], str(tick)+' '+key)
        for key in ('applied_action','consumed_command','controller_record','metrics','reward_terms',
                    'native_interval_summary','servo_receipt','terminal_reason'):
            exact(rows[0]['info'][key], rows[1]['info'][key], str(tick)+' info.'+key)
        if gate:
            exact(rows[0]['action16'], rows[1]['action16'], 'gate-on policy action')
        five = [[next(iterator, sentinel) for _ in range(5)] for iterator in natives]
        c.require(all(row is not sentinel for group in five for row in group), 'native stream shorter than horizon')
        for data, row, group in zip((baseline,gated), rows, five):
            data['validate'](tick, row, group)
        for index, (first, second) in enumerate(zip(*five)):
            exact(first, second, str(5*tick+index)+' native state/torque/contact')
            native_count += 1
    c.require(all(next(iterator, sentinel) is sentinel for iterator in [*iterators,*natives]), 'extra control/native rows')
    c.require(predictions['baseline'] == controls and predictions['gated'] == controls-off,
              'prediction reduction is not exactly the gate-off count')
    return {'controls_per_arm': controls, 'native_per_arm': native_count, 'gate_off_controls': off,
            'control_predictions': predictions, 'total_predict_calls_including_loader_probe':
            {mode: value+1 for mode,value in predictions.items()}, 'control_predictions_saved': off,
            'exact_initial_and_full_states': True, 'exact_controller_memory_and_native_rows': True}


def pure_reader(release):
    forbidden = {'torch','mujoco','onnxruntime','stable_baselines3','glfw','gymnasium','gym'}
    c.require(not forbidden.intersection(sys.modules), 'saved reader was not launched cold')
    path = c.ROOT/'runtime/d1_b22/research/continuation26/root26/reader26.py'
    base = c.load_module('inference32_frozen_reader26', path, release['inputs'][str(path)])
    # The frozen function mixes physical and per-prediction ledgers. Preserve
    # its physical AST verbatim, and check the NEW prediction ledger separately.
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_ledgers')
    start = next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='calls')
    end = next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='(edges, mailbox)')
    c.require(start < end, 'physical/model ledger AST boundary changed')
    node.name='check_physical_ledgers32'; node.body=node.body[:start]+node.body[end:]
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),base.__dict__)  # noqa: S102 - pinned pure physical checks only
    return base


def model_counts(result, session, expected):
    calls, count = result['model_calls'], result['model_calls']['counts']
    values = {'load':1,'torch_load':3,'predict':expected+1,'save':0,'learn':0,'train':0,'forward':0,
              'evaluate_actions':0,'predict_values':0,'backward':0}
    c.require(calls['limits']==session['model_limits'] and result['policy_predictions']==expected
              and all(count.get(key,{}).get('attempted',0)==count.get(key,{}).get('returned',0)==value
                      for key,value in values.items())
              and count['predict']['rows_attempted']==count['predict']['rows_returned']==expected+32
              and calls['actor_rows']==expected+32 and calls['critic_rows']==0,
              'actual model API/probe/control ledger differs')


def saved_mode(release, mode, base):
    import numpy as np
    run = Path(release['plan']['output'])/mode
    session = c.read_json(run/'session.json')
    worker = base.saved_document(run/'worker_receipt.json')
    result = worker['result']
    expected = {'execution_contract_id':'C32_B22_inference_pair_v1','mode':'script','render':False,
                'profile':'flat_1p6','actor':'B','seed':181202,'mirror':False,'control_limit':1600,
                'inference_mode32':mode,'retry_permitted':False,'qualification':False,
                'inference_helper32':str(c.ROOT/'scripts/b22_inference32.py')}
    c.require(all(session.get(k)==v for k,v in expected.items())
              and session['output_directory']==str(run) and worker['session_identity']==c.identity(run/'session.json')
              and worker['execution_contract_id']==session['execution_contract_id'] and worker['arm']==session['arm']==mode
              and worker['failure'] is None and worker['warnings']==[] and worker['archive_failed'] is False
              and worker['writer_partial']==[] and result['completed_controls']==1600
              and result['terminated'] is False and result['truncated'] is True
              and result['inference_mode32']==mode and result['control_step_caller_verified'] is True,
              'new32 complete worker/session identity differs')
    c.require(all(session['source_hashes'].get(k)==v for k,v in release['inputs'].items()), 'worker source closure differs')
    c.require(session['source_hashes'].get(str(Path(release['plan']['output'])/'release.json'))
              == c.identity(Path(release['plan']['output'])/'release.json'), 'worker release identity differs')
    base.check_physical_ledgers32(result,session,1600)
    construction=base.saved_document(run/'construction_receipt.json')
    c.require(construction['proof']['passed'] is True and construction['proof']['dso_path']==session['library']
              and construction['proof']['dso_sha256']==release['inputs'][session['library']]['sha256']
              and len(construction['proof']['jump_slots'])==4
              and all(x['passed'] is True for x in construction['proof']['jump_slots']), 'native construction proof differs')
    binding=base.check_binding(construction['geometry_binding'])
    geometry=base.geometry_map(construction['compiled_geometry'],binding)
    folder=run/'episode_0'
    reset=base.saved_document(folder/'reset.json')
    initial,states=base.saved_arrays(folder/'initial_state.npz'),base.saved_arrays(folder/'states.npz')
    c.require(set(initial)==set(states)==set(STATE_KEYS), 'complete state field set differs')
    base.check_reset(reset['control_reset_state'],initial)
    c.require(reset['seed']==181202, 'reset seed differs')
    segment=base.saved_document(folder/'segment_receipt.json')['native_segment']
    c.require(segment==result['segment_receipts'][0] and len(result['segment_receipts'])==1
              and segment['record_valid'] is True and segment['native_attempted']==segment['native_returned']==8000
              and segment['failure'] is None and segment['archive_failure'] is None
              and segment['partial_native_interval'] is False, 'native segment incomplete')
    names=segment['native_files']
    c.require(len(names)==len(set(names)) and segment['archive_block_manifests']==[name+'.manifest.json' for name in names]
              and all(Path(name).name==name for name in names), 'native archive order/path differs')
    for name in names: base.committed(folder/name)
    base.committed(folder/'controls.jsonl.gz')
    origins=base.saved_document(run/'runtime_module_origins.json')
    c.require(origins.get('run_gui23')==str(c.ROOT/'scripts/run_b22_worker32.py')
              and all(name in session['source_hashes'] for name in origins.values()), 'runtime module origin left closure')
    load=base.saved_document(run/'strict_load.json')
    helper_path=c.ROOT/'scripts/b22_inference32.py'
    helper=c.load_module('inference32_read_proof',helper_path,release['inputs'][str(helper_path)])
    helper.check_stateless_proof32(session['checkpoint_manifest'],load)
    pi,z,previous,stopped,vx=np.zeros(4),0.,None,False,0.
    def validate(tick,row,five):
        nonlocal pi,z,previous,stopped,vx
        raw={'forward_velocity_mps':1.6 if 175<=tick<895 else 0.,'lateral_velocity_mps':0.,
             'yaw_rate_rps':0.,'clearance_m':.455,'jump_requested':False}
        exact(row['raw_command'],raw,'fixed raw script')
        gate=175<=tick<895
        c.require(row['control_stages18']['calculation']['action_enabled'] is gate, 'raw request gate differs')
        vx+=float(np.clip(raw['forward_velocity_mps']-vx,-.005,.005))
        consumed=row['info']['consumed_command']
        c.require(consumed['forward_velocity_mps']==vx and consumed['yaw_rate_rps']==0., 'servo time evolution differs')
        adapter=row['control_stages18']
        after=False if vx!=0. else (True if previous is not None and previous!=0. else stopped)
        c.require(adapter['previous_servo_forward_mps']==previous and adapter['stop_latched_before'] is stopped
                  and adapter['stop_latched_after'] is after and adapter['test_actor_probe'] is False,
                  'stop latch update differs')
        calc=base.check_controller(row,tick,pre_qpos=states['qpos'][tick],pre_qvel=states['qvel'][tick],
            pre_body_forward_mps=float(base._body_com_velocity(states['qpos'][tick],states['qvel'][tick],binding)[0]),
            servo_yaw_rps=0.)
        exact(calc['calculation']['wheel_integral_before_nm'],pi.tolist(),'PI carry')
        exact(calc['calculation']['wheel_common_reference_z_before_rad_s'],z,'z carry')
        pi,z=np.asarray(calc['calculation']['wheel_integral_after_nm']),calc['calculation']['wheel_common_reference_z_after_rad_s']
        previous,stopped=vx,after
        base.check_native(row,five,calc,binding,geometry,5*tick)
    return {'initial':initial,'states':states,'rows':lambda:base.saved_rows(folder/'controls.jsonl.gz'),
            'native':lambda:itertools.chain.from_iterable(base.saved_rows(folder/name) for name in names),
            'validate':validate,'result':result,'session':session,'worker_receipt':c.identity(run/'worker_receipt.json'),
            'geometry':construction['compiled_geometry'],'binding':construction['geometry_binding']}


def read_pair(release_path):
    release_path=Path(release_path).resolve()
    release=c.validate_release(release_path)
    base=pure_reader(release)
    baseline,gated=(saved_mode(release,mode,base) for mode in ('baseline','gated'))
    exact(baseline['geometry'],gated['geometry'],'compiled geometry')
    exact(baseline['binding'],gated['binding'],'compiled binding')
    comparison=compare_numeric(baseline,gated,1600)
    c.require(comparison['gate_off_controls']==880 and comparison['control_predictions']=={'baseline':1600,'gated':720},
              'fixed gate/call saving differs')
    for mode,data in (('baseline',baseline),('gated',gated)):
        model_counts(data['result'],data['session'],comparison['control_predictions'][mode])
    c.validate_release(release_path)
    return {'schema':'b22-inference32-comparison-v1','status':'PASS','release_identity':c.identity(release_path),
            'checkpoint_sha256':c.MODEL_SHA,'comparison':comparison,
            'actual_counts':{'controls':3200,'normal_native':16000,'compiler_native':4},
            'worker_receipts':{mode:data['worker_receipt'] for mode,data in (('baseline',baseline),('gated',gated))},
            'behavior_equivalence':'PASS','qualification_granted':False,'wall_clock_speedup_claimed':False,
            'scope':'one new same-source flat_1p6 B22 seed181202 complete baseline/gated pair; no training'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(argv)
    c.require(not args.report.exists(),'readback report already exists')
    try:
        report=read_pair(args.release)
    except Exception as exc:  # noqa: BLE001 - persist every readback failure without executing anything
        report={'schema':'b22-inference32-comparison-v1','status':'FAIL','error':type(exc).__name__+': '+str(exc),
                'behavior_equivalence':'NOT_VALIDATED','qualification_granted':False}
    c.write_json(args.report,report)
    print(json.dumps(report,allow_nan=False))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
