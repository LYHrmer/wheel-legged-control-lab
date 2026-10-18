"""Synthetic saved-record and inverse-source checks; no model or physics calls."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('reader42_under_test',ROOT/'scripts/read_b22_tracking_reward42.py')
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


def original_terms():
    return {'tracking_vx':1.8,'tracking_vy':.9,'tracking_yaw':.9,'height':-.01,
        'attitude':-.02,'actual_torque':-.03,'action_change':-.04,'leg_speed':-.05,'termination':0.}


def row_for(arm):
    original=original_terms();terms=dict(original);alpha=r.ARMS[arm]
    if alpha!=1.:
        terms['tracking_vx']=2.+alpha*(original['tracking_vx']-2.)
        terms['tracking_yaw']=1.+alpha*(original['tracking_yaw']-1.)
    return {'reward':sum(terms.values()),'info':{'reward_schema':r.REWARD_SCHEMA,
        'training_reward42':{'schema':r.REWARD_SCHEMA,'reward_arm':arm,'alpha':alpha},
        'original_reward_terms42':original,'reward_terms':terms}}


@pytest.mark.parametrize('arm',['control','deficit8'])
def test_only_two_deficits_change_and_input_saved_record_is_untouched(arm):
    row=row_for(arm);before=copy.deepcopy(row);original=original_terms()
    expected=r.reward_expected42(row,original,arm)
    assert row==before and original==original_terms()
    assert all(expected[k]==original[k] for k in r.REWARD_NAMES if k not in ('tracking_vx','tracking_yaw'))
    if arm=='control':assert expected is original
    else:
        assert expected['tracking_vx']==2.+8.*(1.8-2.)
        assert expected['tracking_yaw']==1.+8.*(.9-1.)
        assert expected['tracking_vx']!=8.*1.8


@pytest.mark.parametrize('fault',['arm','alpha','stamp','extra_stamp','schema','original','other_term','total','nan','bool','missing'])
def test_self_consistent_labels_cannot_hide_changed_saved_rewards(fault):
    row=row_for('deficit8');info=row['info']
    if fault=='arm':info['training_reward42']['reward_arm']='control'
    elif fault=='alpha':info['training_reward42']['alpha']=1.
    elif fault=='stamp':info['training_reward42']['schema']='legacy'
    elif fault=='extra_stamp':info['training_reward42']['unchecked']=True
    elif fault=='schema':info['reward_schema']='old_reward'
    elif fault=='original':info['original_reward_terms42']['height']+=.01
    elif fault=='other_term':info['reward_terms']['leg_speed']+=.1;row['reward']+=.1
    elif fault=='total':row['reward']+=.1
    elif fault=='nan':info['reward_terms']['tracking_vx']=float('nan')
    elif fault=='bool':info['reward_terms']['termination']=False
    else:del info['reward_terms']['actual_torque']
    with pytest.raises(ValueError,match='C42'):r.reward_expected42(row,original_terms(),'deficit8')


def test_preregistered_first_block_field_sets_match_frozen_numeric_writer():
    tree=ast.parse((ROOT/'runtime/d1_b22/research/continuation20/curriculum20.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Curriculum20')
    numeric=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_numeric_row')
    row=next(n.value for n in ast.walk(numeric) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='row' for t in n.targets))
    assert set(r.NUMERIC_FIELDS)=={k.value for k in row.keys}|{'stage_'+k for k in r.STAGE_FIELDS}
    gauss=next(n for n in ast.walk(cls) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='append' and ast.unparse(n.func.value)=='self._gaussian')
    assert set(r.GAUSSIAN_FIELDS)=={k.value for k in gauss.args[0].keys}
    old=ast.parse((ROOT/'runtime/d1_b22/research/course_impl08/rl16_curriculum_08.py').read_text())
    stages=next(n.value for n in old.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='STAGE_ARRAYS' for t in n.targets))
    assert set(r.STAGE_FIELDS)==set(ast.literal_eval(stages))


@pytest.fixture
def pairing_data():
    np=pytest.importorskip('numpy')
    numeric={k:np.zeros((1024,1),dtype=np.float64) for k in r.NUMERIC_FIELDS}
    gaussian={k:np.zeros((1024,1),dtype=np.float64) for k in r.GAUSSIAN_FIELDS}
    initial={k:np.zeros(1,dtype=np.float64) for k in r.INITIAL_FIELDS}
    parent={'stage_seed':201011,'stage':2,'arm':'B','actor_reinitialized':False,'optimizer_reinitialized':False,
        'parent_model_identity':{'bytes':1,'sha256':'a'*64},'parent_metadata_identity':{'bytes':1,'sha256':'b'*64},
        'parent_hashes':dict.fromkeys(('policy_state','optimizer_state','actor_mean_state','log_std_state'),'c'*64),
        'parent_num_timesteps':32768,'parent_n_updates':128,'parent_optimizer_steps':1408}
    session={'stage':2,'seed':201011,'arm':'B'}
    return numeric,gaussian,initial,parent,session


def test_reward_fields_alone_differ_while_every_nonreward_hash_pairs(pairing_data):
    numeric,gauss,_initial,_parent,_session=pairing_data
    first=r.pairing42(*pairing_data);first_reward=r.reward_block_identity42(numeric)
    numeric['reward']+=.5;numeric['reward_terms9']+=.5
    assert r.pairing42(*pairing_data)==first
    assert r.reward_block_identity42(numeric)!=first_reward
    assert set(first)=={'schema','initial_state','initial_learning_state','first1024_numeric','first1024_gaussian','seed','course_arm'}
    assert set(first['first1024_numeric'])==set(r.NUMERIC_FIELDS)-{'reward','reward_terms9'}
    gauss['raw_gaussian_action16'][0,0]=.01
    assert r.pairing42(*pairing_data)!=first


@pytest.mark.parametrize('fault',['unknown','missing','short','dtype','initial','seed'])
def test_pairing_does_not_ignore_unknown_numeric_fields_or_state_changes(pairing_data,fault):
    numeric,_gauss,initial,_parent,session=pairing_data
    old=r.pairing42(*pairing_data)
    if fault=='unknown':numeric['returns']=numeric['reward'].copy()
    elif fault=='missing':del numeric['stage_safe_torque_nm']
    elif fault=='short':numeric['control_index']=numeric['control_index'][:1023]
    elif fault=='dtype':numeric['input_observation99']=numeric['input_observation99'].astype('float32')
    elif fault=='initial':initial['observation'][0]=.1
    else:session['seed']=201012
    if fault in ('dtype','initial'):assert r.pairing42(*pairing_data)!=old
    else:
        with pytest.raises(ValueError,match='C42'):r.pairing42(*pairing_data)


@pytest.mark.parametrize('kind',list(r.SOURCES))
def test_actual_frozen_reader_derivations_reverse_exactly_and_keep_old_sha_literals(kind):
    tree,proof=r.adapted_reader42(kind)
    old=ast.parse((ROOT/r.SOURCES[kind][0]).read_text())
    literals=lambda x:[n.value for n in ast.walk(x) if isinstance(n,ast.Constant)
        and isinstance(n.value,str) and len(n.value)==64]
    assert proof['inverse_ast_exact'] and literals(tree)==literals(old)
    if kind=='training':
        f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_check_numeric_reward')
        assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_reward_expected42'
            for n in ast.walk(f))==1
        old_f=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==f.name)
        f.body=[n for n in f.body if not (isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Name) and n.value.func.id=='_reward_expected42')]
        assert ast.dump(f,include_attributes=False)==ast.dump(old_f,include_attributes=False)


@pytest.mark.parametrize('arm',['control','deficit8'])
def test_real_original_numeric_reward_checker_uses_the_new_alpha_seam(arm):
    np=pytest.importorskip('numpy');tree,_=r.adapted_reader42('training')
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_check_numeric_reward')
    row=row_for(arm);info=row['info'];command={'forward_velocity_mps':0.,'lateral_velocity_mps':0.,
        'yaw_rate_rps':0.,'clearance_m':.455}
    fields=('body_com_vx_mps','body_com_vy_mps','body_yaw_rate_rps','x_m','y_m','clearance_m',
        'relative_roll_rad','relative_pitch_rad','task_roll_error_rad','task_pitch_error_rad')
    metrics=dict.fromkeys(fields,0.);metrics.update(body_com_vx_mps=.1,body_yaw_rate_rps=.2,clearance_m=.455)
    original=dict.fromkeys(r.REWARD_NAMES,0.);original.update(tracking_vx=2*math.exp(-(.1/.25)**2),
        tracking_vy=1.,tracking_yaw=math.exp(-(.2/.4)**2))
    info.update(applied_action=[0.]*16,metrics=metrics,raw_operator_command=command,
        consumed_command=command,native_interval_summary={'nonwheel_contact_count':0},original_reward_terms42=original)
    info['reward_terms']=dict(original)
    if arm=='deficit8':
        info['reward_terms'].update(tracking_vx=2+8*(original['tracking_vx']-2),tracking_yaw=1+8*(original['tracking_yaw']-1))
    row.update(reward=sum(info['reward_terms'].values()),terminated=False,
        post_state={'qvel':[0.]*22},native_actuator_traces=[{'applied_nm':[0.]*16,'delayed_nm':[0.]*16} for _ in range(5)])
    calc=dict.fromkeys(r.STAGE_FIELDS,0.)
    numeric={'reward_terms9':np.array([[info['reward_terms'][k] for k in r.REWARD_NAMES]]),
        'raw_command_vx_vy_yaw_clearance':np.array([[command[k] for k in command]]),
        'servo_command_vx_vy_yaw_clearance':np.array([[command[k] for k in command]]),
        'actuator_delayed5x16':np.zeros((1,5,16)),'actuator_applied5x16':np.zeros((1,5,16)),
        'native_nonwheel_contacts':np.zeros(1),**{'stage_'+k:np.zeros(1) for k in r.STAGE_FIELDS}}
    for key,field in (('body_com_vx_mps','body_com_vx_mps'),('body_com_vy_mps','body_com_vy_mps'),
        ('body_yaw_rate_rps','body_yaw_rate_rps'),('base_x_m','x_m'),('base_y_m','y_m'),('clearance_m','clearance_m'),
        ('geometry_relative_roll_rad','relative_roll_rad'),('geometry_relative_pitch_rad','relative_pitch_rad'),
        ('task_roll_error_rad','task_roll_error_rad'),('task_pitch_error_rad','task_pitch_error_rad')):
        numeric[key]=np.array([metrics[field]])
    namespace={'np':np,'math':math,'LEGS':tuple(i for i in range(16) if i%4!=3),'TORQUE_LIMIT':np.ones(16),
        'require':r.require,'close':lambda x,y,atol=1e-6:bool(np.allclose(x,y,rtol=0,atol=atol)),
        'REWARD_NAMES':r.REWARD_NAMES,'STAGE_FIELDS':r.STAGE_FIELDS,
        '_reward_expected42':lambda row,expected:r.reward_expected42(row,expected,arm)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual-frozen-reward-checker>','exec'),namespace)  # noqa: S102 -- execute the pinned pure numeric checker, without model or simulator code
    result=namespace['_check_numeric_reward'](row,numeric,0,calc,{'dof_addresses':list(range(16))},np.zeros(16))
    assert np.array_equal(result,np.zeros(16))
    if arm=='deficit8':
        namespace['_reward_expected42']=lambda row,expected:expected
        with pytest.raises(ValueError):namespace['_check_numeric_reward'](row,numeric,0,calc,{'dof_addresses':list(range(16))},np.zeros(16))


def test_counter_preserves_real_call_return_limits_and_restores_on_error():
    class Servo:
        def advance(self,tick,value):return tick+value
    geometry=SimpleNamespace(FullDriveCommandServo=Servo)
    with r.geometry_counter42(geometry,2) as counts:
        servo=geometry.FullDriveCommandServo()
        assert servo.advance(1,4)==5 and servo.advance(2,5)==7
        with pytest.raises(ValueError,match='budget'):servo.advance(3,6)
        assert counts=={'attempted':2,'returned':2}
    assert geometry.FullDriveCommandServo is Servo
    with pytest.raises(RuntimeError),r.geometry_counter42(geometry,2):raise RuntimeError('synthetic interruption')
    assert geometry.FullDriveCommandServo is Servo


def test_default_cli_stays_cold_even_if_parent_pytest_has_framework_modules():
    code='import importlib.util,sys;from pathlib import Path;p=Path("scripts/read_b22_tracking_reward42.py");s=importlib.util.spec_from_file_location("cold42",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert m.main([])==0;assert not any(n.partition(".")[0] in m.FORBIDDEN|{"numpy"} for n in sys.modules)'
    process=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
    assert json.loads(process.stdout)['status']=='PLAN_ONLY'


@pytest.mark.parametrize('fault',['preloaded','preload_env','library_env'])
def test_saved_execution_rejects_framework_or_loader_injection_before_math(monkeypatch,fault):
    if fault=='preloaded':monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    else:monkeypatch.setenv('LD_PRELOAD' if fault=='preload_env' else 'LD_LIBRARY_PATH','/untrusted')
    with pytest.raises(ValueError,match='C42 reader refuses'):r.assert_saved_only42()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True))


def checkpoint_manifest_from_frozen_save42(folder):
    """Use the real save expression's separate metadata SHA and three payloads."""
    path=ROOT/'runtime/d1_b22/research/continuation21/checkpoint21.py'
    assert r.identity(path)['sha256']=='ef23653bd0419a8fd10d12d05325b8d2ad932ff0426175e60eab96b86aa0019f'
    function=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)
        and n.name=='save_final_and_verify')
    loop=next(n for n in function.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple))
    assert [n.id for n in loop.iter.elts]==['MODEL_FILE','PROBE_OBS_FILE','PROBE_ACTION_FILE']
    files={name:r.identity(folder/name) for name in r.CHECKPOINT_PAYLOADS42}
    original=next(n.value for n in function.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='manifest' for t in n.targets))
    selected=[(key,value) for key,value in zip(original.keys,original.values,strict=True)
        if ast.literal_eval(key) in ('folder','files','metadata_sha256')]
    expression=ast.Expression(ast.Dict(keys=[x[0] for x in selected],values=[x[1] for x in selected]))
    namespace={'os':os,'folder':str(folder),'files':files,'payload':(folder/'final_metadata.json').read_bytes(),
        '_hash_bytes':lambda payload:r.hashlib.sha256(payload).hexdigest()}
    # Only the three actual expressions extracted from SHA-pinned save source.
    return eval(compile(ast.fix_missing_locations(expression),'<frozen-checkpoint-manifest-shape>','eval'),{'__builtins__':{}},namespace)


def write_checkpoint42(run):
    folder=run/'final_checkpoint';folder.mkdir(parents=True)
    for name in ('final_model.zip','final_metadata.json','final_probe_observations_f32.bin','final_probe_actions_f32.bin'):
        (folder/name).write_bytes(b'synthetic, not model weights')
    manifest=checkpoint_manifest_from_frozen_save42(folder)
    write(run/'final_checkpoint_manifest.json',manifest)
    return run,folder,manifest


def checkpoint(tmp_path):return write_checkpoint42(tmp_path/'train')


def test_final_input_map_is_exact_and_rejects_foreign_or_changed_members(tmp_path):
    run,folder,manifest=checkpoint(tmp_path);mapping=r.final_checkpoint_inputs42(run)
    assert len(manifest['files'])==3 and 'final_metadata.json' not in manifest['files']
    assert set(mapping)=={str(run/'final_checkpoint_manifest.json'),*(str(p) for p in folder.iterdir())}
    (folder/'final_model.zip').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes differ'):r.final_checkpoint_inputs42(run)
    manifest['files']['final_model.zip']=r.identity(folder/'final_model.zip')
    write(run/'final_checkpoint_manifest.json',manifest);assert r.final_checkpoint_inputs42(run)!=mapping
    (folder/'unexpected.bin').write_bytes(b'new')
    with pytest.raises(ValueError,match='member set'):r.final_checkpoint_inputs42(run)


@pytest.mark.parametrize('fault',['missing','changed','incorrect_four_payload_manifest'])
def test_final_metadata_is_separately_bound_by_real_save_manifest(tmp_path,fault):
    run,folder,manifest=checkpoint(tmp_path)
    if fault=='missing':(folder/'final_metadata.json').unlink()
    elif fault=='changed':(folder/'final_metadata.json').write_bytes(b'altered metadata')
    else:
        manifest['files']['final_metadata.json']=r.identity(folder/'final_metadata.json')
        write(run/'final_checkpoint_manifest.json',manifest)
    with pytest.raises(ValueError,match='member set|bytes differ'):r.final_checkpoint_inputs42(run)


@pytest.mark.parametrize('alter_metadata',[False,True])
def test_actual_eval_load_bridge_uses_separate_metadata_sha_without_fourth_payload(tmp_path,alter_metadata):
    run=tmp_path/'eval';run.mkdir();manifests={}
    for arm in r.ARMS:
        _,folder,_=write_checkpoint42(tmp_path/('train_'+arm))
        write(folder/'final_metadata.json',{'training_receipt':{'final_hashes':{'policy_state':'a'*64}}})
        manifest=checkpoint_manifest_from_frozen_save42(folder);manifests[arm]=manifest
        write(run/(arm+'_load.json'),{'loader_schema':'d1-b22-tracking42-final-checkpoint-v1',
            'metadata_sha256':manifest['metadata_sha256'],'probe_rows':32,'probe_actions_byte_exact':True,
            'matches_training_final_policy_state':True,'verified_without_engine_or_reset':True,
            'loaded_policy_state_sha256':'a'*64})
    if alter_metadata:
        write(tmp_path/'train_control/final_checkpoint/final_metadata.json',
            {'training_receipt':{'final_hashes':{'policy_state':'b'*64}}})
    function=next(n for n in ast.parse(Path(r.__file__).read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name=='read_evaluation42')
    loop=next(n for n in function.body if isinstance(n,ast.For))
    namespace={'context':SimpleNamespace(session={'eval_manifests':manifests}),'run':run,'ARMS':r.ARMS,
        'reader':SimpleNamespace(document=r.read_json),'hashes':{},'loads':{},'Path':Path,
        'identity':r.identity,'read_json':r.read_json,'require':r.require}
    compiled=compile(ast.Module(body=[loop],type_ignores=[]),'<actual-eval-final-load-bridge>','exec')
    if alter_metadata:
        with pytest.raises(ValueError,match='checkpoint bytes differ'):exec(compiled,namespace)  # noqa: S102 -- the production metadata-only bridge, no model/predict/physics calls
    else:
        exec(compiled,namespace)  # noqa: S102 -- the production metadata-only bridge, no model/predict/physics calls
        assert set(namespace['hashes'])==set(r.ARMS) and len(namespace['loads'])==2


@pytest.fixture
def saved_stage(tmp_path):
    output=tmp_path/'execution';run=output/'train_control';run.mkdir(parents=True)
    worker_source=tmp_path/'worker.py';worker_source.write_text('# synthetic worker source, never executed')
    release_path=tmp_path/'release.json';write(release_path,{'fake':'independent release'})
    stage={'name':'train_control','mode':'train','reward_arm':'control','alpha':1,'controls':32768,
        'normal_native':163840,'compiler_native':2,'hard_s':1200,'quota_bytes':3<<30}
    release={'source':{'fake_source':True},'inputs':r.required_inputs42()}
    session={'schema':'b22-tracking42-session-v1','execution_contract_id':r.CONTRACT,
        'release_path':str(release_path),'release_identity':r.identity(release_path),'worker_identity':r.identity(worker_source),
        'output_directory':str(run),'parent_pid':7,'nonce':'a'*48,'wall_clock_utc':'2026-10-18T00:00:00+00:00',
        'host_started_monotonic':0.,'retry_permitted':False,'source_hashes':copy.deepcopy(release['inputs']),
        'mode':'train','reward_arm':'control','alpha':1,'arm':'B','quota_bytes':3<<30}
    write(run/'session.json',session)
    counts={'controls':32768,'normal_native':163840,'compiler_native':2}
    worker={'schema':'b22-tracking42-worker-receipt-v1','status':'COMPLETE','execution_contract_id':r.CONTRACT,
        'session_identity':r.identity(run/'session.json'),'execution_complete':True,'failure':None,'cleanup_errors':[],
        'warnings':[],'archive_failed':False,'stop_latched':False,'control_step_caller_verified':True,
        'arm':'B','mode':'train','reward_arm':'control','alpha':1,'actual_counts':copy.deepcopy(counts),
        'known_partial_counts':copy.deepcopy(counts),'C_final':{'construction_attempts':2,'construction_returns':2,
            'control_attempts':163840,'control_returns':163840,'phase':0,'target_model':0,'target_data':0,'violations':0},
        'python':{'control_attempted':32768,'control_completed':32768,'native_attempted':163840,
            'native_returned':163840,'clock_advanced_substeps':163840,'native_failed':0,'forbidden_entries':0},
        'native_guard':{'native_attempted':163840,'native_returned':163840,'native_checked':163840,'failure':None,'segments':[{}]},
        'quota42':{'strict_worker_before_write':True,'exhausted':False,'limit_bytes':3<<30}}
    write(run/'worker_receipt.json',worker)
    supervision={'schema':'b22-tracking42-worker-supervision-v1','stage':'train_control',
        'release_identity':r.identity(release_path),'session_identity':r.identity(run/'session.json'),
        'worker_identity':r.identity(worker_source),'exit_code':0,'failure':None,'supervision_returned':True,
        'cleanup':{'remaining':{}},'elapsed_s':2.}
    write(run/'supervisor_receipt.json',supervision)
    parent_spec=importlib.util.spec_from_file_location('pure_parent_counts42',ROOT/'scripts/run_b22_tracking_reward42.py')
    parent=importlib.util.module_from_spec(parent_spec);parent_spec.loader.exec_module(parent)
    api=SimpleNamespace(OUTPUT=output,WORKER=worker_source,
        validate_release42=lambda path:copy.deepcopy(release),stage_by_name42=lambda name:copy.deepcopy(stage),
        make_session42=lambda *args,**kwargs:copy.deepcopy(session),checked_counts42=parent.checked_counts42,
        directory_bytes42=lambda path:sum(p.stat().st_size for p in path.rglob('*') if p.is_file()))
    return SimpleNamespace(output=output,run=run,worker=worker,session=session,supervision=supervision,
        path=release_path,api=api,release=release,stage=stage)


def test_actual_parent_count_gate_and_new_supervisor_schema_join_saved_closure(saved_stage):
    context=r.checked_stage42(saved_stage.path,'train_control',saved_stage.api)
    assert context.actual=={'controls':32768,'normal_native':163840,'compiler_native':2}
    closure=r.training_closure42(context)
    assert closure(context.run,context.session,context.worker,context.supervisor)==context.sources
    with pytest.raises(ValueError,match='foreign closure'):
        closure(context.run,context.session,dict(context.worker,status='old'),context.supervisor)


@pytest.mark.parametrize('fault',['legacy_worker','partial','bool_count','native_count','unknown_count','foreign_session',
    'legacy_supervisor','dirty_cleanup','source','late','quota'])
def test_saved_closure_rejects_typed_partial_foreign_dirty_or_changed_evidence(saved_stage,fault):
    s=saved_stage
    if fault=='legacy_worker':s.worker['schema']='d1-coverage-worker-20-v1'
    elif fault=='partial':s.worker['status']='INCOMPLETE'
    elif fault=='bool_count':s.worker['actual_counts']['compiler_native']=True
    elif fault=='native_count':s.worker['C_final']['control_returns']-=1
    elif fault=='unknown_count':s.worker['actual_counts']=None
    elif fault=='foreign_session':s.worker['session_identity']['sha256']='e'*64
    elif fault=='legacy_supervisor':s.supervision['schema']='old-host'
    elif fault=='dirty_cleanup':s.supervision['cleanup']['remaining']={'9':7}
    elif fault=='source':s.session['source_hashes'][next(iter(s.session['source_hashes']))]['sha256']='d'*64
    elif fault=='late':s.supervision['elapsed_s']=1231.
    else:s.worker['quota42']['exhausted']=True
    write(s.run/'worker_receipt.json',s.worker);write(s.run/'supervisor_receipt.json',s.supervision)
    write(s.run/'session.json',s.session)
    with pytest.raises((ValueError,TypeError)):r.checked_stage42(s.path,'train_control',s.api)


def test_read_stage_claims_once_counts_real_fake_servo_and_does_not_construct_any_model(saved_stage,monkeypatch):
    s=saved_stage
    monkeypatch.setattr(r,'assert_saved_only42',lambda:None)  # Test process may contain prior collector's framework imports.
    class Servo:
        def advance(self,tick,command):return command
    geometry=SimpleNamespace(FullDriveCommandServo=Servo)
    def factory(arm,closure):
        assert arm=='control'
        closure(s.run,s.session,s.worker,s.supervision)
        return SimpleNamespace(geometry=geometry,proofs={'synthetic':True})
    def read(context,maths):
        maths.geometry.FullDriveCommandServo().advance(0,'same returned command')
        return {'training_valid':True,'proof_pure_servo_intervals':1}
    monkeypatch.setattr(r,'read_training42',read)
    value=r.read_stage42(s.path,'train_control',api=s.api,math_factory=factory)
    assert value['status']=='COMPLETE' and value['actual_pure_servo_advances']==1
    assert value['reader_model_calls']==value['reader_physics_calls']==0
    assert geometry.FullDriveCommandServo is Servo
    with pytest.raises(ValueError,match='exclusive'):r.read_stage42(s.path,'train_control',api=s.api,math_factory=factory)


def test_failed_saved_read_attempt_is_not_silently_retried(saved_stage,monkeypatch):
    s=saved_stage;monkeypatch.setattr(r,'assert_saved_only42',lambda:None)
    class Servo:
        def advance(self,*args):raise RuntimeError('synthetic data error')
    geometry=SimpleNamespace(FullDriveCommandServo=Servo)
    factory=lambda *args:SimpleNamespace(geometry=geometry,proofs={})
    monkeypatch.setattr(r,'read_training42',lambda context,maths:maths.geometry.FullDriveCommandServo().advance())
    with pytest.raises(RuntimeError,match='data error'):r.read_stage42(s.path,'train_control',api=s.api,math_factory=factory)
    assert geometry.FullDriveCommandServo is Servo and (s.run/'saved_reader42_attempt.json').is_file()
    with pytest.raises(FileExistsError):r.read_stage42(s.path,'train_control',api=s.api,math_factory=factory)


@pytest.fixture
def actual_pair_score(monkeypatch):
    pytest.importorskip('numpy')
    monkeypatch.syspath_prepend(str(ROOT/'runtime/d1_b22/research/course_impl08'))
    path=ROOT/'runtime/d1_b22/research/continuation18/score18.py'
    spec=importlib.util.spec_from_file_location('actual_c18_for42',path)
    c18=importlib.util.module_from_spec(spec);spec.loader.exec_module(c18)
    tree,_=r.adapted_reader42('score')
    schema={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SCORE_SCHEMA','PAIR_SCHEMA')}
    namespace={'c18':c18,'ORIGINAL_SIX':tuple(c18.HELDOUT_CASES),'Sequence':list,'Mapping':dict,'Any':object,**schema}
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score_pairs')
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<frozen-real-C18-pair-gates>','exec'),namespace)  # noqa: S102 -- frozen pure score identity adapter only
    return SimpleNamespace(ORIGINAL_SIX=tuple(c18.HELDOUT_CASES),score_pairs=namespace['score_pairs'],
        SCORE_SCHEMA=schema['SCORE_SCHEMA'],cases=c18.HELDOUT_CASES,windows=c18.case_windows)


def scored_records(score):
    rows=[]
    for case_id,(terrain,_speed,_yaw,seed,cap) in score.cases.items():
        begin,end=score.windows(case_id)['drive'];ticks=end-begin
        for actor in ('zero','control','deficit8'):
            sse=.8 if actor=='deficit8' else 1.
            rows.append({'schema':score.SCORE_SCHEMA,'case_id':case_id,'actor':'zero' if actor=='zero' else 'final_policy',
                'experiment_actor':actor,'seed':seed,'terrain':terrain,'task_passed':True,'task_reasons':[],
                'completed_controls':cap,'windows':{'control_cap':cap},'record':{'record_valid':True,
                    'full_horizon':True,'early_task_termination':False},'safety':{'passed':True},
                'rl_terms':{'sse_total':sse,'sse_vx':sse*.8,'sse_yaw':sse*.2,'drive_window':[begin,end],
                    'drive_completed_stop':end,'drive_completed_ticks':ticks,'drive_complete':True,
                    'torque_cost_sum':.1*ticks,'torque_cost_by_tick':[.1]*ticks}})
    return rows


def comparison_inputs(score):
    training={arm:{'training_valid':True,'pairing42':{'same_actual_nonreward':'fixture'},
        'first1024_reward_arrays42':{'different_reward':arm},'coverage_valid':arm=='control',
        'coverage':{'separately_reported':True}} for arm in r.ARMS}
    evaluation={'evaluation_valid':True,'floors_screened_arms':False,'all_three_actors_evaluated':True,
        'case_records':scored_records(score),'floor_records':{a:{'numeric_gate_passed':False} for a in r.ARMS}}
    return training,evaluation


def test_real_original_six_gate_not_floor_or_training_valid_decides_contribution(actual_pair_score):
    training,evaluation=comparison_inputs(actual_pair_score)
    value=r.comparison_math42(training,evaluation,actual_pair_score)
    assert value['RL_contribution_passed_by_arm']=={'control':False,'deficit8':True}
    assert not value['qualification_granted'] and not value['floors_screened_arms']
    assert value['actor_case_totals']['deficit8']['case_denominator']==6
    assert value['training_coverage']['deficit8']['coverage_valid'] is False
    bad=next(x for x in evaluation['case_records'] if x['experiment_actor']=='deficit8')
    bad['rl_terms'].update(sse_total=5.,sse_vx=4.,sse_yaw=1.)
    assert r.comparison_math42(training,evaluation,actual_pair_score)['RL_contribution_passed_by_arm']['deficit8'] is False


@pytest.mark.parametrize('fault',['pair','missing','duplicate','filter'])
def test_comparison_requires_exact_pair_and_complete_unscreened_matrix(actual_pair_score,fault):
    training,evaluation=comparison_inputs(actual_pair_score)
    if fault=='pair':training['deficit8']['pairing42']['same_actual_nonreward']='changed'
    elif fault=='missing':evaluation['case_records'].pop()
    elif fault=='duplicate':evaluation['case_records'][-1]=copy.deepcopy(evaluation['case_records'][0])
    else:evaluation['floors_screened_arms']=True
    with pytest.raises(ValueError,match='C42'):r.comparison_math42(training,evaluation,actual_pair_score)


@pytest.fixture
def saved_comparison(tmp_path,actual_pair_score):
    output=tmp_path/'execution';output.mkdir();plan=tmp_path/'release.json';write(plan,{'fake':'release'})
    release={'source':{'same_source':True}};training,evaluation=comparison_inputs(actual_pair_score)
    parent_spec=importlib.util.spec_from_file_location('pure_parent_for_summary42',ROOT/'scripts/run_b22_tracking_reward42.py')
    parent=importlib.util.module_from_spec(parent_spec);parent_spec.loader.exec_module(parent)
    write(output/'bootstrap/worker_receipt.json',{'synthetic':'bootstrap zero execution'})
    records=[{'name':'bootstrap','path':'bootstrap/worker_receipt.json',
        'report':r.identity(output/'bootstrap/worker_receipt.json'),'independent_saved_reader_executed':False}]
    for name in ('train_control','train_deficit8','eval'):
        run=output/name;run.mkdir();write(run/'session.json',{'synthetic':name});write(run/'worker_receipt.json',{'synthetic':name})
        mode='eval' if name=='eval' else 'train';n=30600 if mode=='eval' else 32768
        detail=evaluation if mode=='eval' else training[name.removeprefix('train_')]
        value={'schema':'b22-tracking42-readback-v1','status':'COMPLETE','execution_contract_id':r.CONTRACT,
            'stage':name,'release_identity':r.identity(plan),'reader_identity':r.identity(Path(r.__file__)),
            'session_identity':r.identity(run/'session.json'),'worker_receipt_identity':r.identity(run/'worker_receipt.json'),
            'actual_counts':{'controls':n,'normal_native':n*5,'compiler_native':2},'actual_pure_servo_advances':1,
            'reader_model_calls':0,'reader_physics_calls':0,**copy.deepcopy(detail)}
        if mode=='train':
            write_checkpoint42(run)
            value['final_checkpoint_inputs42']=r.final_checkpoint_inputs42(run)
        value['artifacts']=r.artifact_inputs42(run)
        path=output/(name+'_readback.json');write(path,value);records.append({'name':name,'report':r.identity(path)})
    write(output/'report.json',{'schema':'b22-tracking42-supervision-v1','execution_contract_id':r.CONTRACT,
        'release_identity':r.identity(plan),'source':release['source'],'stages':records})
    api=SimpleNamespace(OUTPUT=output,validate_release42=lambda path:copy.deepcopy(release),
        stage_by_name42=lambda name:{'mode':'eval' if name=='eval' else 'train','controls':30600 if name=='eval' else 32768,
            'normal_native':153000 if name=='eval' else 163840,'compiler_native':2},
        checked_counts42=parent.checked_counts42)
    return SimpleNamespace(output=output,path=plan,api=api,score=actual_pair_score)


def test_comparison_reads_only_accepted_summaries_no_geometry_or_inference(saved_comparison,monkeypatch):
    s=saved_comparison;monkeypatch.setattr(r,'assert_saved_only42',lambda:None)
    monkeypatch.setattr(r,'load_math42',lambda *args:pytest.fail('geometry readback was rerun'))
    monkeypatch.setattr(r,'geometry_counter42',lambda *args:pytest.fail('servo was replayed'))
    value=r.compare42(s.path,api=s.api,score_loader=lambda:(s.score,{'synthetic_score_proof':True}))
    assert value['status']=='COMPLETE' and value['actual_pure_servo_advances']==0
    assert set(value['stage_reports'])=={'train_control','train_deficit8','eval'}
    assert value['RL_contribution_passed_by_arm']=={'control':False,'deficit8':True}
    with pytest.raises(ValueError,match='exclusive'):r.compare42(s.path,api=s.api,score_loader=lambda:(s.score,{}))


@pytest.mark.parametrize('fault',['reader_label','identity','extra_readback'])
def test_comparison_distinguishes_bootstrap_receipt_from_three_independent_readbacks(saved_comparison,monkeypatch,fault):
    s=saved_comparison;monkeypatch.setattr(r,'assert_saved_only42',lambda:None)
    path=s.output/'report.json';journal=json.loads(path.read_text());bootstrap=journal['stages'][0]
    if fault=='reader_label':bootstrap['independent_saved_reader_executed']=True
    elif fault=='identity':bootstrap['report']['sha256']='0'*64
    else:journal['stages'].append(copy.deepcopy(journal['stages'][-1]))
    write(path,journal)
    with pytest.raises(ValueError,match='bootstrap|three readbacks'):
        r.compare42(s.path,api=s.api,score_loader=lambda:(s.score,{}))


@pytest.mark.parametrize('fault',['accepted_summary','model'])
def test_comparison_rejects_changed_accepted_score_or_final_model(saved_comparison,monkeypatch,fault):
    s=saved_comparison;monkeypatch.setattr(r,'assert_saved_only42',lambda:None)
    if fault=='accepted_summary':
        path=s.output/'eval_readback.json';value=json.loads(path.read_text());value['case_records'][0]['rl_terms']['sse_total']=900.
        write(path,value)  # All raw payload hashes remain unchanged: parent acceptance SHA is essential.
    else:(s.output/'train_control/final_checkpoint/final_model.zip').write_bytes(b'different model')
    with pytest.raises(ValueError,match='changed|differs'):
        r.compare42(s.path,api=s.api,score_loader=lambda:(s.score,{}))


def test_local_pure_legacy_constructor_uses_actual_frozen_schema_without_any_model():
    manifest=ROOT/'runtime/d1_b22/runtime_manifest_v27.json'
    if not (ROOT/'results/d1_main_route_execution_20260930_22/spec22.json').is_file() or not manifest.is_file():
        pytest.skip('optional local frozen-runtime constructor; synthetic/source regressions remain active')
    configured=json.loads(manifest.read_text())['runtime_environment']['PYTHONPATH']
    directories=[Path(value) for value in configured.split(os.pathsep) if value]
    if not directories or not all(path.is_dir() for path in directories):
        pytest.skip('optional constructor requires every fixed local manifest import directory; CI runs all synthetic/source regressions')
    env={k:v for k,v in os.environ.items() if k not in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH')}
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    code='import importlib.util,sys;from pathlib import Path;p=Path("scripts/read_b22_tracking_reward42.py");s=importlib.util.spec_from_file_location("cold42_math",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);g=m.load_math42("control",lambda *args:{});assert g.training.CONTRACT_ID==m.CONTRACT;assert g.eval.RECORD_SCHEMA=="d1-b22-tracking42-heldout-record-v1";assert g.score.SCORE_SCHEMA=="d1-b22-tracking42-case-score-v1";assert g.recipes.T["ppo_seed_by_stage"]==[201001,201011];assert not any(n.partition(".")[0] in m.FORBIDDEN for n in sys.modules);print("PURE_CONSTRUCTOR_PASS")'
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='PURE_CONSTRUCTOR_PASS'
