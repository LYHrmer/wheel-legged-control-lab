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
O = W / 'run_03'
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

MODEL_SHA='4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e'
receipt=read(O/'continuation_receipt.json');outer=read(O/'root_launcher_receipt.json')
assert outer['exit_code']==0 and outer['error'] is None and not outer['post_execution_hash_mismatches']
assert receipt['new_five_case_execution_valid'] and receipt['actual_counts_consistent']
assert receipt['original_run_02_study_qualified'] is False and receipt['original_final_C_counts'] is None
assert receipt['new_training_controls']==0
stitched=read(O/'eight_case_stitch.json')
assert len(stitched['cases'])==8
freeze=read(W/'root_execution_freeze_03.json')
for name,row in freeze['files'].items():
    p=Path(name);assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
initial=read(O/'runtime_initial.json')
assert initial['binding_proof']['passed']
assert all(initial['initial_C_state_after_arm'][k]==0 for k in ['control_attempts','control_returns','construction_attempts','construction_returns','ccd_attempts','ccd_returns','violations'])
case_results={}
with np.load(W/'run_02/evaluation/speed200_box_zero/states.npz',allow_pickle=False) as old_initial:
    paired={(.2,'box'):{k:old_initial[k][0].copy() for k in ('qpos','qvel','ctrl','qacc_warmstart','observation')}}
for case in stitched['cases'][3:]:
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
        assert t['checkpoint_sha256']==MODEL_SHA
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
actual=sum(v['controls'] for v in case_results.values())
assert actual<=6000 and receipt['actual_new_evaluation_controls']==actual
assert receipt['C_state']['control_attempts']==receipt['C_state']['control_returns']==5*actual
assert receipt['C_state']['construction_attempts']==receipt['C_state']['construction_returns']==5
assert receipt['native_monitor']['checked_native_returns']==receipt['native_monitor']['passed_native_returns']==5*actual
previous_after=None
for name,row in case_results.items():
    before=read(O/'evaluation'/f'{name}.before.json');after=read(O/'evaluation'/f'{name}.after.json')
    assert before['error'] is None and after['error'] is None
    for key in ('control_attempts','control_returns'):
        assert after['C_state'][key]-before['C_state'][key]==5*row['controls']
        if previous_after is not None:assert before['C_state'][key]==previous_after['C_state'][key]
    for key in ('control_attempted','control_completed'):
        assert after['python_ledger'][key]-before['python_ledger'][key]==row['controls']
    assert after['C_state']['phase']==0 and after['C_state']['target_model']==after['C_state']['target_data']==0
    previous_after=after
prefix={}
for filename,count in [('endpoints.jsonl.gz',578),('trace.jsonl.gz',577),('native.jsonl.gz',2888)]:
    old=iter(rows(W/'run_02/evaluation/speed200_box_final_policy'/filename));new=iter(rows(O/'evaluation/speed200_box_final_policy'/filename))
    differences=[]
    for i in range(count):
        a,b=next(old),next(new)
        if a!=b:differences.append(i)
    prefix[filename]={'compared_rows':count,'all_equal':not differences,'first_differing_index':None if not differences else differences[0]}
manifest={str(p.relative_to(O)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file()}
write(W/'run_03_archive_manifest_01.json',{'scope':'all closed continuation run_03 files','files':manifest})
write(W/'root_run_03_readback_01.json',{'passed':True,'scope':'offline new five-case native/state/torque/action/hash/C-boundary readback; no engine or policy imports',
    'frozen_inputs':len(freeze['files']),'new_archive_files':len(manifest),'new_case_records':case_results,
    'actual_new_controls':actual,'actual_new_normal_mj_step':5*actual,'actual_new_compiler_mj_step':5,
    'interrupted_prefix_comparison':prefix,'original_run_02_qualified':False,'original_final_C_counts':None,
    'aggregate_policy_task_and_speed_gates_passed':receipt['aggregate_four_policy_task_and_speed_gates_passed'],
    'uninterrupted_global_accounting':False,'causal_rl_improvement_claimed':False})
print(json.dumps({'new_five_case_readback_passed':True,'new_controls':actual,'new_normal_mj_step':5*actual,
                  'all_saved_prefix_rows_equal':all(v['all_equal'] for v in prefix.values()),
                  'aggregate_policy_gates':receipt['aggregate_four_policy_task_and_speed_gates_passed']}))
