"""Offline independent raw-record reconciliation; never imports engine or policy."""
import gzip
import hashlib
import importlib.abc
import json
import math
import sys
from pathlib import Path

class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'torch', 'stable_baselines3', 'gymnasium'}:
            raise RuntimeError('Forbidden engine/policy import: ' + fullname)

sys.meta_path.insert(0, NoEngine())
import numpy as np

W = Path(__file__).resolve().parent
O = W / 'run_02'
def read(p): return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def rows(p):
    with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
        for line in f: yield json.loads(line)
def same(a,b): return np.array_equal(np.asarray(a), np.asarray(b))
def write(p, j):
    with p.open('x') as f: json.dump(j, f, indent=2, allow_nan=False)

observation=read(W/'run_02_interruption_observation_01.json')
assert not (O/'study_receipt.json').exists() and not (O/'root_launcher_receipt.json').exists()
cases=[]
for name in ('speed200_plane_zero','speed200_plane_final_policy','speed200_box_zero'):
    folder=O/'evaluation'/name
    actual=read(folder/'receipt.json')
    assert actual['error'] is None and actual['completed_control_intervals']==1200
    cases.append({'case':name,'speed_mps':.2,'terrain':'box' if '_box_' in name else 'plane',
                  'actor':'final_policy' if name.endswith('final_policy') else 'zero',
                  'score':read(folder/'score.json')})
receipt={'training':read(O/'training/training_receipt.json'),'evaluation':{'cases':cases}}
freeze = read(W/'root_execution_freeze_02.json')
for name, spec in freeze['files'].items():
    p=Path(name); assert p.stat().st_size == spec['bytes'] and sha(p)==spec['sha256'], name
training = receipt['training']
assert training['physical_control_transitions']==65536
updates = list(rows(O/'training/completed_updates.jsonl'))
assert len(updates)==512
for i,u in enumerate(updates,1):
    assert (u['train_call'],u['completed_control'],u['optimization_epochs'])==(i,128*i,4*i)
    assert u['optimizer_state_entries']>0
assert updates[-1]['policy_sha256']==training['final_parameter_sha256']
assert updates[-1]['actor_sha256']==training['final_actor_sha256']
assert training['initial_parameter_sha256']!=training['final_parameter_sha256']
assert sha(O/'training/final_checkpoint/model.zip')==training['model_sha256']
assert sha(O/'training/final_checkpoint/model.metadata.json')==training['checkpoint_metadata_sha256']
samples = iter(rows(O/'training/gaussian_samples.jsonl'))
train_count=reset_count=0
train_action_abs=0.0
for row in rows(O/'training/controls.jsonl'):
    if row['event']=='episode_reset': reset_count+=1; continue
    train_count+=1
    sample=next(samples)
    assert row['transition']==sample['transition']==train_count
    assert math.isclose(sum(row['reward_terms'].values()),row['reward'],abs_tol=1e-10)
    scalar=sample['actual_policy_action']
    assert scalar==sample['clipped_env_input']==max(-1,min(1,sample['preclip_gaussian_action']))
    active=row['raw_command']['forward_velocity_mps']!=0
    expected=[.25*scalar if active else 0.0]*4+[0.0]*4
    assert same(expected,row['applied_action']) and same(expected,sample['applied_physical_action'])
    train_action_abs=max(train_action_abs,max(abs(v) for v in expected))
assert next(samples,None) is None and train_count==65536
case_results={}
paired={}
for case in receipt['evaluation']['cases']:
    name=case['case']; folder=O/'evaluation'/name
    traces=list(rows(folder/'trace.jsonl.gz')); endpoints=list(rows(folder/'endpoints.jsonl.gz'))
    states=np.load(folder/'states.npz',allow_pickle=False)
    score=read(folder/'score.json'); count=len(traces)
    assert score==case['score'] and score['record_valid']
    assert count==score['metrics']['completed_control_intervals'] and len(endpoints)==count+1
    pair=(case['speed_mps'],case['terrain'])
    initial={k:states[k][0].copy() for k in ('qpos','qvel','ctrl','qacc_warmstart','observation')}
    if case['actor']=='zero': paired[pair]=initial
    else:
        for k in initial: assert same(initial[k],paired[pair][k]),(name,k)
    for tick,t in enumerate(traces):
        assert t['tick']==tick and t['endpoint_tick']==tick+1
        cmd=case['speed_mps'] if 175<=tick<975 else 0.0
        assert t['raw_command']['forward_velocity_mps']==cmd
        scalar=max(-1,min(1,t['policy_action'][0])); leg=.25*scalar if cmd else 0.0
        assert same(t['action'],[leg]*4+[0.0]*4) and same(t['action'],t['applied_action'])
        assert same(t['controller_result']['clipped_action'],t['action'])
        assert same(t['torque_nm'],t['controller_result']['torque_nm'])
        assert same(t['torque_nm'],t['stop_record']['safe_torque_nm'])
        assert t['stop_active']==(tick>=975) and not t['turn_active']
        assert t['checkpoint_sha256']==training['model_sha256']
    native_count=box_steps=nonwheel=drive_all_wheels_unloaded=0; last=None
    unloaded_length=max_unloaded_length=0; unloaded_start=max_unloaded_start=None
    for i,n in enumerate(rows(folder/'native.jsonl.gz')):
        assert n['index']==i and n['returned'] and n['error'] is None
        assert abs(n['actual_dt_s']-.002)<1e-12
        tick=i//5
        assert same(n['ctrl_nm'],traces[tick]['torque_nm'])
        if last is not None:
            assert same(last['qpos_returned'],n['qpos_before'])
            assert same(last['qvel_returned'],n['qvel_before'])
        if i%5==0:
            assert same(states['qpos'][tick],n['qpos_before'])
            assert same(states['qvel'][tick],n['qvel_before'])
        if i%5==4:
            assert same(states['qpos'][tick+1],n['qpos_returned'])
            assert same(states['qvel'][tick+1],n['qvel_returned'])
        nonwheel+=n['contacts']['nonwheel_terrain_contacts']
        box_steps+=any(v>0 for v in n['contacts']['wheel_box_positive_normal_load_n'])
        unloaded=(175<=tick<975 and all(v<=0 for v in n['contacts']['wheel_positive_normal_load_n']))
        if unloaded:
            drive_all_wheels_unloaded+=1
            if unloaded_length==0: unloaded_start=i
            unloaded_length+=1
            if unloaded_length>max_unloaded_length:
                max_unloaded_length=unloaded_length; max_unloaded_start=unloaded_start
        else: unloaded_length=0; unloaded_start=None
        native_count+=1; last=n
    assert native_count==5*count
    events=list(rows(folder/'native_entry_events.jsonl.gz'))
    assert len(events)==native_count and all(e['attempt']==i for i,e in enumerate(events))
    assert nonwheel==score['metrics']['nonwheel_contact_native_total']
    assert case['terrain']!='box' or box_steps==score['metrics']['positive_wheel_box_native_steps']
    if count>=875:
        vx=np.array([e['body_vx_mps'] for e in endpoints[275:876]])
        assert abs(vx.mean()-score['metrics']['speed_window_mean_body_vx_mps'])<1e-12
        assert abs(np.sqrt(np.mean((vx-case['speed_mps'])**2))-score['metrics']['speed_window_rms_command_error_mps'])<1e-12
    case_results[name]={'controls':count,'native':native_count,'positive_box_native_steps':int(box_steps),
                        'drive_native_steps_without_any_positive_wheel_load':int(drive_all_wheels_unloaded),
                        'longest_drive_unloaded_native_steps':int(max_unloaded_length),
                        'longest_drive_unloaded_duration_s':max_unloaded_length*.002,
                        'longest_drive_unloaded_start_index':max_unloaded_start,
                        'longest_drive_unloaded_end_index':None if max_unloaded_start is None else max_unloaded_start+max_unloaded_length-1,
                        'task_passed':score['task_passed'],'record_valid':score['record_valid'],
                        'paired_initial_verified':case['actor']=='final_policy'}
verified=65536+sum(c['controls'] for c in case_results.values())
assert verified==69136
manifest={str(p.relative_to(O)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file()}
write(W/'run_02_interrupted_archive_manifest_01.json',{'scope':'all retained interrupted run_02 files including partial gzip; not a normal process closure','files':manifest})
write(W/'root_run_02_interrupted_readback_01.json',{'passed_for_completed_records_only':True,
    'scope':'offline raw training/action/update/hash and three closed evaluation native-chain/control-torque records; no engine or policy imports',
    'freeze_inputs':len(freeze['files']),'retained_archive_files':len(manifest),'training_controls':train_count,
    'training_resets':reset_count,'completed_updates':len(updates),'max_abs_training_applied_leg_normalized':train_action_abs,
    'completed_cases':case_results,'verified_training_and_closed_eval_controls':verified,
    'saved_complete_control_return_lower_bound':verified+577,
    'saved_normal_native_return_lower_bound':5*verified+2888,
    'native_C_final_counters_available':False,'process_final_exit_status_available':False,
    'full_reservation_consumed':{'control':75136,'normal_mj_step':375680,'compiler_mj_step':5},
    'study_qualified':False,'causal_rl_improvement_claimed':False,
    'partial_fourth_case_not_reclassified':True,'interruption_observation_sha256':sha(W/'run_02_interruption_observation_01.json')})
print(json.dumps({'completed_records_readback_passed':True,'retained_files':len(manifest),'verified_closed_record_controls':verified,'overall_qualification':False}))
