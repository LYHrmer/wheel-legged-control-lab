"""Saved-only C42 training and fixed-task readback; never loads an actor or engine."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
CONTRACT='C45_B22_rough_terrain_curriculum_arm_v1'
REWARD_SCHEMA='d1-b22-training-tracking-deficit-alpha42-v1'
ARMS={'ffrough':8.}
REWARD_NAMES=('tracking_vx','tracking_vy','tracking_yaw','height','attitude',
              'actual_torque','action_change','leg_speed','termination')
STAGE_FIELDS=(
    'consumed_joint_position_rad','consumed_joint_velocity_rad_s','consumed_nominal_joint_target_rad',
    'consumed_nominal_wheel_speed_rad_s','consumed_support_torque_nm','consumed_base_rotation_world_from_body',
    'consumed_foot_jacobian_world','consumed_wheel_omega_rad_s','geometric_joint_target_rad',
    'rate_limited_joint_target_rad','position_limited_joint_target_rad','joint_target_rad',
    'wheel_speed_target_rad_s','wheel_speed_error_rad_s','wheel_integral_before_nm',
    'wheel_integral_candidate_nm','wheel_integral_after_nm','leg_pd_nm','base_wheel_torque_nm',
    'base_request_torque_nm','projected_jx','relative_forward_mps','leg_damping_delta_torque_nm',
    'common_wheel_delta_torque_nm','final_request_torque_nm','safe_torque_nm')
NUMERIC_FIELDS=(
    'control_index','episode_index','episode_tick','input_observation99','policy_env_input16',
    'policy_clipped16','effective_action16','raw_command_vx_vy_yaw_clearance',
    'servo_command_vx_vy_yaw_clearance','reward_terms9','reward','terminated','truncated',
    'body_com_vx_mps','body_com_vy_mps','body_yaw_rate_rps','base_x_m','base_y_m','clearance_m',
    'geometry_relative_roll_rad','geometry_relative_pitch_rad','task_roll_error_rad',
    'task_pitch_error_rad','native_nonwheel_contacts','actuator_delayed5x16','actuator_applied5x16',
    *(f'stage_{name}' for name in STAGE_FIELDS))
GAUSSIAN_FIELDS=('control_index','episode_index','episode_tick','raw_gaussian_action16',
                 'clipped_action16','effective_action16')
INITIAL_FIELDS=('qpos','qvel','ctrl','qacc_warmstart','observation','time')
CHECKPOINT_PAYLOADS42=('final_model.zip','final_probe_observations_f32.bin','final_probe_actions_f32.bin')
SOURCES={
    'training':('runtime/d1_b22/research/continuation22/read_training22.py','39291ddeeb9951b4d078d8d60de81eaa0f51a3bf796bf9464bb5dfe997ffa1bd'),
    'eval':('runtime/d1_b22/research/continuation22/read_eval22.py','edda8aa55b183d65c78b6c9ee016987232e137e2075326cc7f5c19d0593d055b'),
    'score':('runtime/d1_b22/research/continuation22/score22.py','aa00850fa9b6bf239067225e822da24ba449a8c772e362e44aba1b69afca71cd')}
FORBIDDEN={'mujoco','glfw','torch','stable_baselines3','gym','gymnasium','engine_binding',
           'wheel_legged_control','cv2','jax','tensorflow'}


def require(condition,message):
    if not condition:raise ValueError(message)


def identity(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'C42 regular saved/source file required: '+str(path))
    digest,size=hashlib.sha256(),0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block);size+=len(block)
    return {'bytes':size,'sha256':digest.hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))


def required_inputs42():
    values={}
    for name,digest in SOURCES.values():
        path=ROOT/name;entry=identity(path)
        require(entry['sha256']==digest,'C42 saved-reader original source changed: '+name)
        values[str(path)]=entry
    return values


def reward_expected42(row,original_expected,reward_arm):
    """Join independently computed original terms to the two actual C42 deficits.

    The original training reader computes ``original_expected`` from saved COM,
    commands, native torques and actions. Its other controller/numeric joins
    remain in that same pass; this function performs no inference or replay.
    """
    require(reward_arm in ARMS,'C42 unknown reward arm')
    alpha=ARMS[reward_arm];info=row['info']
    stamp=info.get('training_reward42');original=info.get('original_reward_terms42')
    terms=info.get('reward_terms')
    require(isinstance(stamp,dict) and set(stamp)=={'schema','alpha','reward_arm'}
        and stamp.get('schema')==REWARD_SCHEMA and stamp.get('reward_arm')==reward_arm
        and type(stamp.get('alpha')) in (int,float) and stamp['alpha']==alpha
        and info.get('reward_schema')==REWARD_SCHEMA,'C42 training reward identity differs')
    for values in (original_expected,original,terms):
        require(isinstance(values,dict) and set(values)==set(REWARD_NAMES)
            and all(type(v) in (int,float) and math.isfinite(v) for v in values.values()),
            'C42 nine finite reward terms required')
    require(all(math.isclose(original[k],original_expected[k],rel_tol=0,abs_tol=1e-8)
        for k in REWARD_NAMES),'C42 original reward differs from independent physical calculation')
    expected=original_expected
    if alpha!=1.:
        expected={**original_expected,
            'tracking_vx':2.+alpha*(original_expected['tracking_vx']-2.),
            'tracking_yaw':1.+alpha*(original_expected['tracking_yaw']-1.)}
    require(all(math.isclose(terms[k],expected[k],rel_tol=0,abs_tol=1e-8) for k in REWARD_NAMES)
        and type(row.get('reward')) in (int,float) and math.isfinite(row['reward'])
        and math.isclose(row['reward'],sum(expected.values()),rel_tol=0,abs_tol=1e-8),
        'C42 actual tracking deficits or reward total differ')
    return expected


def array_identity42(value):
    import numpy as np
    require(isinstance(value,np.ndarray) and value.dtype.kind in 'biuf' and np.isfinite(value).all(),
            'C42 finite saved numeric array required')
    return {'dtype':value.dtype.str,'shape':list(value.shape),
            'sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def pairing42(numeric0,gaussian0,initial,parentage,session):
    """Hash first saved blocks already checked by the original one-pass reader."""
    require(set(numeric0)==set(NUMERIC_FIELDS) and set(gaussian0)==set(GAUSSIAN_FIELDS)
        and set(initial)==set(INITIAL_FIELDS),'C42 preregistered first-block/initial array schema differs')
    require(all(value.ndim>=1 and value.shape[0]==1024 for values in (numeric0,gaussian0)
        for value in values.values()),'C42 complete first1024 blocks required')
    require(type(session.get('stage')) is int and session['stage']==2
        and type(session.get('seed')) is int and session['seed']==201011
        and session.get('arm')=='B','C42 stage2 seed/course pairing differs')
    require(parentage.get('stage_seed')==201011 and parentage.get('arm')=='B'
        and parentage.get('stage')==2 and parentage.get('actor_reinitialized') is False
        and parentage.get('optimizer_reinitialized') is False,'C42 parentage pairing differs')
    keys=('parent_model_identity','parent_metadata_identity','parent_hashes',
          'parent_num_timesteps','parent_n_updates','parent_optimizer_steps')
    require(all(k in parentage for k in keys) and set(parentage['parent_hashes'])=={
        'policy_state','optimizer_state','actor_mean_state','log_std_state'},'C42 complete initial learning state missing')
    return {'schema':'b22-tracking42-pairing-v1','initial_state':{k:array_identity42(initial[k]) for k in INITIAL_FIELDS},
        'initial_learning_state':copy.deepcopy({k:parentage[k] for k in keys}),
        'first1024_numeric':{k:array_identity42(numeric0[k]) for k in NUMERIC_FIELDS if k not in ('reward','reward_terms9')},
        'first1024_gaussian':{k:array_identity42(gaussian0[k]) for k in GAUSSIAN_FIELDS},
        'seed':201011,'course_arm':'B'}


def reward_block_identity42(numeric0):
    """Reward differences remain outside the non-reward pairing identity."""
    return {k:array_identity42(numeric0[k]) for k in ('reward','reward_terms9')}


def adapted_reader42(kind):
    """Inverse-check the finite identity, import, reward and result seams."""
    require(kind in SOURCES,'C42 unknown reader derivation')
    required_inputs42();path=ROOT/SOURCES[kind][0]
    original=ast.parse(path.read_text());tree=copy.deepcopy(original);edits=[]
    def put(parent,field,index,new,label):
        old=getattr(parent,field)[index] if index is not None else getattr(parent,field)
        edits.append((parent,field,index,old,label,False))
        if index is None:setattr(parent,field,new)
        else:getattr(parent,field)[index]=new
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module in ('recipes22','read_eval22'):
            put(node,'module',None,{'recipes22':'b22_recipes42','read_eval22':'b22_read_eval42'}[node.module],
                'private pure42 import')
        if kind=='eval' and isinstance(node,ast.Import) and any(n.name=='score22' for n in node.names):
            require(len(node.names)==1,'C42 score module import seam changed')
            put(node,'names',None,[ast.alias(name='b22_score42',asname='score22')],'private42 score math, actual new actor labels')
        if kind=='eval' and isinstance(node,ast.ImportFrom) and node.module=='heldout22':
            require([x.name for x in node.names]==['PAIR_FIELDS','RECORD_SCHEMA'],'C42 raw recorder fields changed')
            put(tree,'body',tree.body.index(node),ast.parse(
                "PAIR_FIELDS,RECORD_SCHEMA=('qpos','qvel','ctrl','qacc_warmstart','observation'),'d1-b22-tracking42-heldout-record-v1'"
            ).body[0],'explicit42 raw identity, no recorder/framework import')
        if kind=='training' and isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCHEMA' for t in node.targets):
            put(node,'value',None,ast.Constant('b22-tracking42-full-training-readback-v1'),'honest42 detailed training identity')
        if kind=='score' and isinstance(node,ast.Assign):
            schemas={'SCORE_SCHEMA':'d1-b22-tracking42-case-score-v1','PAIR_SCHEMA':'d1-b22-tracking42-pair-score-v1',
                     'FLOOR_SCORE_SCHEMA':'d1-b22-tracking42-floor-score-v1'}
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in schemas:
                    put(node,'value',None,ast.Constant(schemas[target.id]),'honest42 scored-result identity')
                if isinstance(target,ast.Name) and target.id=='SPEC':
                    put(node,'value',None,ast.Name('_spec42',ast.Load()),'fixed original spec data path, no new numeric values')
    if kind=='score':
        for name,old,new in (
            ('score_case',('zero','old','A','B'),('zero','control','deficit8')),
            ('score_pairs',('A','B'),('control','deficit8')),
            ('score_floor_canonical',('A','B'),('control','deficit8'))):
            function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            changed=0
            for node in ast.walk(function):
                if isinstance(node,ast.Tuple) and all(isinstance(x,ast.Constant) for x in node.elts) and tuple(x.value for x in node.elts)==old:
                    put(node,'elts',None,[ast.Constant(x) for x in new],'new42 actor enum only');changed+=1
            require(changed==1,'C42 exact actor-enum seam changed: '+name)
    if kind=='training':
        reward=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_check_numeric_reward')
        expected=next(n for n in reward.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets))
        at=reward.body.index(expected)+1
        reward.body.insert(at,ast.parse('expected=_reward_expected42(row,expected)').body[0])
        edits.append((reward,'body',at,None,'only two independently recomputed tracking deficits',True))
        verify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_training')
        stages=0;host_paths=0
        for node in ast.walk(verify):
            if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='stage' and ast.unparse(node)=='stage == 1':
                put(node,'comparators',0,ast.Constant(2),'actual same-course stage2');stages+=1
            if isinstance(node,ast.Constant) and node.value=='host_receipt.json':
                put(node,'value',None,'supervisor_receipt.json','new42 parent-owned supervision receipt');host_paths+=1
        require(stages==host_paths==1,'C42 original training stage/supervision seam changed')
        result=next(n for n in verify.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets))
        value=copy.deepcopy(result.value)
        value.keys.extend([ast.Constant('pairing42'),ast.Constant('first1024_reward_arrays42')])
        value.values.extend([ast.parse('_pairing_ready42(numeric0,gaussian0,session,run)',mode='eval').body,
                             ast.parse('reward_block_identity42(numeric0)',mode='eval').body])
        put(result,'value',None,value,'hash already-checked first blocks; rewards outside pairing')
    result=copy.deepcopy(tree)
    for parent,field,index,old,_label,inserted in reversed(edits):
        if inserted:getattr(parent,field).pop(index)
        elif index is None:setattr(parent,field,old)
        else:getattr(parent,field)[index]=old
    require(ast.dump(tree,include_attributes=False)==ast.dump(original,include_attributes=False),
            'C42 undeclared saved-reader AST change')
    return ast.fix_missing_locations(result),{'source':str(path),'source_sha256':SOURCES[kind][1],
        'derived_ast_sha256':hashlib.sha256(ast.dump(result,include_attributes=False).encode()).hexdigest(),
        'inverse_ast_exact':True,'seams':[x[4] for x in edits],
        'private_namespace_bindings':['CONTRACT_ID=C42','C42 worker/source closure','capture first reset only']}


class _NoExecution42(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.partition('.')[0] in FORBIDDEN:raise RuntimeError('C42 saved reader forbids '+fullname)


def assert_saved_only42():
    require('LD_PRELOAD' not in os.environ and 'LD_LIBRARY_PATH' not in os.environ,
            'C42 reader refuses loader injection environment')
    require(not any(n.partition('.')[0] in FORBIDDEN for n in sys.modules),'C42 reader refuses preloaded model/physics modules')
    maps=Path('/proc/self/maps').read_text().lower()
    require('libmujoco' not in maps and 'libepa01_engine' not in maps,'C42 reader refuses mapped physics library')


def load_file42(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


train45=load_file42('b22_roughtrain45',ROOT/'scripts/b22_roughtrain45.py')


def install_derived_verifier44():
    """Register verify_control18 with the single yaw-law edit before the frozen reader imports it."""
    path=ROOT/'runtime/d1_b22/research/continuation18/verify_control18.py'
    module=ModuleType('verify_control18');module.__file__=str(path);sys.modules['verify_control18']=module
    exec(compile(train45.derived_verifier_source(path.read_text(),train45.FEEDFORWARD),str(path),'exec'),module.__dict__)  # noqa: S102 -- one audited edit


def parent_api42():return load_file42('_b22_tracking42_parent_reader',ROOT/'scripts/run_b22_roughtrain45.py')
def worker_api42():return load_file42('_b22_tracking42_worker_reader',ROOT/'scripts/worker_b22_roughtrain45.py')


@contextmanager
def geometry_counter42(geometry,limit):
    """Count real pure geometry servo calls without changing the original math."""
    original=geometry.FullDriveCommandServo;counts={'attempted':0,'returned':0}
    class CountedServo42(original):
        def advance(self,*args,**kwargs):
            require(counts['attempted']<limit,'C42 saved-reader pure geometry servo budget exhausted')
            counts['attempted']+=1
            value=super().advance(*args,**kwargs)
            counts['returned']+=1
            return value
    geometry.FullDriveCommandServo=CountedServo42
    try:yield counts
    finally:geometry.FullDriveCommandServo=original


def load_math42(reward_arm,worker_closure):
    """Private adapted readers; imports only frozen saved-data math and NumPy."""
    assert_saved_only42();sys.meta_path.insert(0,_NoExecution42())
    worker=worker_api42();config=read_json(worker.MANIFEST)
    paths=[str(ROOT/'runtime/d1_b22/research/continuation22'),
           *(p for p in config['runtime_environment']['PYTHONPATH'].split(os.pathsep) if p),str(ROOT)]
    sys.path[:0]=list(dict.fromkeys(paths))
    spec=read_json(worker.SPEC_PATH);spec['training']['ppo_seed_by_stage']=[201001,201011]
    recipes,recipe_proof=worker._exec_module42('recipes',{'_spec42':spec})
    train45.apply_curriculum(recipes,train45.ARMS[reward_arm]['curriculum'])
    install_derived_verifier44()
    modules={};proofs={'recipes':recipe_proof}
    for kind in ('score','eval','training'):
        tree,proof=adapted_reader42(kind)
        name='b22_score42' if kind=='score' else 'b22_read_'+('eval42' if kind=='eval' else 'training42')
        module=ModuleType(name);module.__file__=str(ROOT/SOURCES[kind][0]);sys.modules[name]=module
        module._spec42=spec
        exec(compile(tree,module.__file__,'exec'),module.__dict__)  # noqa: S102 -- pinned inverse-checked pure saved reader
        module.CONTRACT_ID=CONTRACT;modules[kind]=module;proofs[kind]=proof
    train=modules['training'];captured={}
    original_start=train._check_episode_start
    def start(*args,**kwargs):
        value=original_start(*args,**kwargs)
        if args[1]==0:captured['initial']=value[4]
        return value
    train._check_episode_start=start
    train._worker_closure=worker_closure
    train._reward_expected42=lambda row,expected:reward_expected42(row,expected,reward_arm)
    def ready(numeric,gaussian,session,run):
        require('initial' in captured,'C42 first initial state was not checked')
        return pairing42(numeric,gaussian,captured['initial'],train.document(run/'parentage.json'),session)
    train._pairing_ready42=ready;train.reward_block_identity42=reward_block_identity42
    return SimpleNamespace(training=train,eval=modules['eval'],recipes=recipes,
        score=modules['eval'].score22,geometry=sys.modules['geometry21'],proofs=proofs)


def same_json42(first,second):
    return json.dumps(first,sort_keys=True,allow_nan=False)==json.dumps(second,sort_keys=True,allow_nan=False)


def write_exclusive42(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def artifact_inputs42(run):
    values={}
    for path in sorted(Path(run).rglob('*')):
        require(not path.is_symlink(),'C42 saved tree contains symlink')
        if path.is_file():values[str(path.relative_to(run))]=identity(path)
    return values


def final_checkpoint_inputs42(run):
    manifest_path=run/'final_checkpoint_manifest.json';manifest=read_json(manifest_path)
    folder=run/'final_checkpoint'
    require(Path(manifest['folder'])==folder and folder.is_dir() and not folder.is_symlink(),
            'C42 exact final-only checkpoint folder required')
    members=list(folder.iterdir())
    require(set(manifest['files'])==set(CHECKPOINT_PAYLOADS42)
        and all(p.is_file() and not p.is_symlink() for p in members)
        and {p.name for p in members}=={*CHECKPOINT_PAYLOADS42,'final_metadata.json'},
        'C42 final checkpoint member set differs')
    values={str(manifest_path):identity(manifest_path)}
    for path in members:
        entry=identity(path)
        require((entry['sha256']==manifest['metadata_sha256'] if path.name=='final_metadata.json'
            else entry==manifest['files'][path.name]),'C42 actual final checkpoint bytes differ')
        values[str(path)]=entry
    return values


def checked_stage42(release_path,stage_name,api):
    release=api.validate_release42(Path(release_path))
    stage=api.stage_by_name42(stage_name)
    require(stage['mode'] in ('train','eval'),'C42 saved reader has no bootstrap execution route')
    run=api.OUTPUT/stage_name
    require(run.is_dir() and not any(p.is_symlink() for p in (run,*run.parents)),
            'C42 fixed regular saved run required')
    session=read_json(run/'session.json');worker=read_json(run/'worker_receipt.json')
    supervisor=read_json(run/'supervisor_receipt.json')
    canonical=api.make_session42(stage,release,parent_pid=session['parent_pid'],nonce=session['nonce'],
        started=session['host_started_monotonic'],wall_clock_utc=session['wall_clock_utc'])
    require(same_json42(session,canonical),'C42 saved canonical session differs')
    require(session.get('schema')=='b22-tracking42-session-v1' and session.get('execution_contract_id')==CONTRACT
        and Path(session['output_directory'])==run and Path(session['release_path'])==Path(release_path)
        and session.get('release_identity')==identity(release_path)
        and session.get('worker_identity')==identity(api.WORKER)
        and type(session.get('parent_pid')) is int and session['parent_pid']>0
        and isinstance(session.get('nonce'),str) and len(session['nonce'])==48
        and session.get('retry_permitted') is False,'C42 saved stage/session identity differs')
    require(worker.get('schema')=='b22-tracking42-worker-receipt-v1'
        and worker.get('status')=='COMPLETE' and worker.get('execution_contract_id')==CONTRACT
        and worker.get('session_identity')==identity(run/'session.json')
        and worker.get('execution_complete') is True and worker.get('failure') is None
        and worker.get('cleanup_errors')==[] and worker.get('warnings')==[]
        and worker.get('archive_failed') is False and worker.get('stop_latched') is False
        and worker.get('control_step_caller_verified') is True
        and worker.get('arm')==session['arm']
        and all(same_json42(worker.get(k),session.get(k)) for k in ('mode','reward_arm','alpha')),
        'C42 complete saved worker required')
    actual=api.checked_counts42(worker.get('actual_counts'),stage,complete=True)
    require(same_json42(actual,worker.get('known_partial_counts')),'C42 actual/known saved worker counts differ')
    require(supervisor.get('schema')=='b22-tracking42-worker-supervision-v1'
        and supervisor.get('stage')==stage_name and supervisor.get('release_identity')==identity(release_path)
        and supervisor.get('session_identity')==identity(run/'session.json')
        and supervisor.get('worker_identity')==identity(api.WORKER)
        and type(supervisor.get('exit_code')) is int and supervisor['exit_code']==0
        and supervisor.get('failure') is None and supervisor.get('supervision_returned') is True
        and supervisor.get('cleanup',{}).get('remaining')=={}
        and type(supervisor.get('elapsed_s')) in (int,float) and math.isfinite(supervisor['elapsed_s'])
        and 0<=supervisor['elapsed_s']<=stage['hard_s']+30,'C42 saved supervisor closure differs')
    c,py,ng=(worker[k] for k in ('C_final','python','native_guard'));n=actual['controls']
    require(c['construction_attempts']==c['construction_returns']==2
        and c['control_attempts']==c['control_returns']==5*n
        and c['phase']==c['target_model']==c['target_data']==c['violations']==0
        and py['control_attempted']==py['control_completed']==n
        and py['native_attempted']==py['native_returned']==py['clock_advanced_substeps']==5*n
        and py['native_failed']==py['forbidden_entries']==0
        and ng['native_attempted']==ng['native_returned']==ng['native_checked']==5*n
        and ng['failure'] is None,'C42 saved C/Python/native physical counts differ')
    for name,entry in session['source_hashes'].items():
        require(identity(name)==entry,'C42 saved-stage source or final checkpoint changed: '+name)
    require(all(session['source_hashes'].get(k)==v for k,v in required_inputs42().items()),
            'C42 saved-reader originals absent from frozen closure')
    quota=worker.get('quota42',{})
    require(quota.get('strict_worker_before_write') is True and quota.get('exhausted') is False
        and quota.get('limit_bytes')==session['quota_bytes']
        and api.directory_bytes42(run)<=session['quota_bytes'],'C42 saved stage byte quota differs')
    return SimpleNamespace(release=release,stage=stage,run=run,session=session,worker=worker,
        supervisor=supervisor,actual=actual,sources=session['source_hashes'])


def training_closure42(context):
    def closure(run,session,worker,host):
        require(run==context.run and same_json42(session,context.session) and same_json42(worker,context.worker)
            and same_json42(host,context.supervisor),'C42 training reader received foreign closure')
        require(len(worker['native_guard']['segments'])==1,'C42 training must have exactly one native segment')
        return context.sources
    return closure


def read_training42(context,maths):
    detail=maths.training.verify_training(context.run,context.run/'training_readback42.json')
    final=final_checkpoint_inputs42(context.run)
    metadata=read_json(context.run/'final_checkpoint/final_metadata.json')
    stamp=metadata.get('tracking_reward42',{})
    require(metadata.get('reward_schema')==REWARD_SCHEMA and set(stamp)=={'schema','alpha','reward_arm'}
        and stamp.get('schema')==REWARD_SCHEMA and stamp.get('reward_arm')==context.stage['reward_arm']
        and type(stamp.get('alpha')) in (int,float) and stamp['alpha']==ARMS[context.stage['reward_arm']],
        'C42 final checkpoint reward identity differs')
    return {'training_valid':detail['training_valid'],'archive_valid':detail['archive_valid'],
        'learning_valid':detail['learning_valid'],'coverage_valid':detail['coverage_valid'],'coverage':detail['coverage'],
        'detailed_readback_identity':identity(context.run/'training_readback42.json'),
        'pairing42':detail['pairing42'],'first1024_reward_arrays42':detail['first1024_reward_arrays42'],
        'final_checkpoint_inputs42':final,'actual_learning':detail['actual_learning'],
        'geometry_reset_proofs':detail['geometry_reset_proofs'],
        'proof_pure_servo_intervals':detail['pure_nominal_servo_intervals_verified']}


def checked_eval_calls42(worker,predicted):
    calls=worker['model_calls']['counts']
    expected={'load':2,'torch_load':6,'save':0,'learn':0,'train':0,'forward':0,
        'evaluate_actions':0,'predict_values':0,'backward':0,'predict':2+predicted}
    for key,count in expected.items():
        row=calls.get(key,{})
        require(type(row.get('attempted',0)) is int and type(row.get('returned',0)) is int
            and row.get('attempted',0)==row.get('returned',0)==count,'C42 eval model API count differs: '+key)
    require(worker['model_calls']['actor_rows']==predicted+64 and worker['model_calls']['critic_rows']==0
        and all(row['rows_attempted']==row['rows_returned'] for row in calls.values()),
        'C42 eval actor/critic/model row counts differ')


def read_evaluation42(context,maths):
    import numpy as np
    run=context.run;reader=maths.eval
    construction=reader.document(run/'construction_receipt.json')
    binding,_=maths.training._geometry(construction);geometry=construction['compiled_geometry']
    require(len(maths.recipes.ORDER)==6 and set(maths.recipes.ORDER)==set(maths.score.ORIGINAL_SIX),
            'C42 exact original six task matrix required')
    actors=('zero','control','deficit8');hashes={};loads={}
    for arm in ARMS:
        manifest=context.session['eval_manifests'][arm]
        folder=Path(manifest['folder'])
        require(folder==run.parent/('train_'+arm)/'final_checkpoint','C42 eval must use this arm final only')
        model_id=identity(folder/'final_model.zip');metadata=read_json(folder/'final_metadata.json')
        require(model_id==manifest['files']['final_model.zip']
            and identity(folder/'final_metadata.json')['sha256']==manifest['metadata_sha256'],
                'C42 eval final checkpoint bytes differ')
        hashes[arm]=model_id['sha256'];proof=reader.document(run/(arm+'_load.json'));loads[arm]=proof
        require(proof.get('loader_schema')=='d1-b22-tracking42-final-checkpoint-v1'
            and proof.get('metadata_sha256')==manifest['metadata_sha256']
            and proof.get('probe_rows')==32 and proof.get('probe_actions_byte_exact') is True
            and proof.get('matches_training_final_policy_state') is True
            and proof.get('verified_without_engine_or_reset') is True
            and proof.get('loaded_policy_state_sha256')==metadata['training_receipt']['final_hashes']['policy_state'],
            'C42 actual final checkpoint load/probe provenance differs')
    floors={};floor_summaries=[];floor_initial=[];floor_resets=[]
    floor_spec=('floor_0p4_600','flat',.4,context.session['floor_seed'],600,325,425)
    for arm in ARMS:
        folder=run/'floor'/arm
        summary,canonical=reader.heldout_case(run,floor_spec,arm,binding,geometry,
            expected_variant='combined',folder_override=folder)
        require(canonical['checkpoint_sha256']==hashes[arm],'C42 floor checkpoint differs')
        saved=reader.document(folder/'case_receipt.json')
        require(context.worker['result']['floors'][arm]=={'receipt':saved,'numeric_gate_pending_independent_reader':True},
                'C42 worker floor archive receipt differs')
        floors[arm]=maths.score.score_floor_canonical(canonical,expected_seed=context.session['floor_seed'])
        floor_summaries.append(summary);floor_initial.append(reader.arrays(folder/'initial_state.npz'))
        floor_resets.append(reader._reset_state(reader.document(folder/'reset_receipt.json')))
    require({p.name for p in (run/'floor').iterdir() if p.is_dir()}==set(ARMS)
        and reader._same_arrays(*floor_initial) and same_json42(*floor_resets),
        'C42 two unconditional floors must use the same initial/control state')
    summaries=[];scored=[];saved_cases=[];parsed=[]
    for case_id in maths.recipes.ORDER:
        task=maths.recipes.TASK_BY_ID[case_id]
        spec=(case_id,task.terrain,task.speed_mps,task.seed,task.horizon,task.hold_start,task.release_tick)
        parsed.append((spec,False,actors))
        for actor in actors:
            summary,canonical=reader.heldout_case(run,spec,actor,binding,geometry,expected_variant='combined')
            require(canonical['checkpoint_sha256']==hashes.get(actor),'C42 actual case checkpoint differs')
            score=maths.score.score_case(canonical,expected_seed=task.seed,mirror=False)
            scored.append(dict(score,experiment_actor=actor,checkpoint_sha256=hashes.get(actor)))
            summaries.append(summary)
            raw=reader.document(run/'heldout'/(case_id+'_'+actor)/'case_receipt.json')
            saved_cases.append({k:raw[k] for k in ('case_id','actor','seed','completed_controls',
                'record_valid','terminated','truncated','stop_reason','policy_predictions','checkpoint_sha256')})
    require({p.name for p in (run/'heldout').iterdir() if p.is_dir()}=={
        case+'_'+actor for case in maths.recipes.ORDER for actor in actors},'C42 omitted/unplanned eval folder')
    pairs=reader._new_pairs(run,parsed,list(ARMS))
    observed=sum(x['completed_controls'] for x in (*floor_summaries,*summaries))
    predicted=sum(x['predict_calls'] for x in (*floor_summaries,*summaries))
    require(observed==context.actual['controls']==context.worker['result']['completed_controls']
        and context.worker['result']['enabled_arms']==list(ARMS)
        and context.worker['result']['cases']==saved_cases
        and context.worker['result']['full_comparison_executed'] is True
        and context.worker['result']['evaluation_reward_unchanged'] is True
        and type(context.worker['result']['evaluation_gain']) is int
        and context.worker['result']['evaluation_gain']==1,'C42 complete unconditional evaluation matrix/counts differ')
    checked_eval_calls42(context.worker,predicted)
    require(all(np.isfinite(x['rl_terms']['sse_total']) for x in scored),'C42 finite saved task metric required')
    return {'evaluation_valid':True,'case_records':scored,'case_integrity':summaries,'pairing_records':pairs,
        'floor_records':floors,'floor_integrity':floor_summaries,'floor_pair_initial_equal':True,
        'floors_screened_arms':False,'all_three_actors_evaluated':True,'scored_cases':18,
        'complete_original_six_by_actor':{a:all(x['completed_controls']==x['windows']['control_cap']
            for x in scored if x['experiment_actor']==a) for a in actors},
        'actual_eval_model_calls':context.worker['model_calls'],'actual_load_proofs':loads,
        'proof_pure_servo_intervals':sum(x['geometry21']['pure_servo_intervals'] for x in (*floor_summaries,*summaries))}


def read_stage42(release_path,stage_name,report_path=None,*,api=None,math_factory=load_math42):
    assert_saved_only42();api=api or parent_api42()
    context=checked_stage42(Path(release_path),stage_name,api)
    report_path=Path(report_path or api.OUTPUT/(stage_name+'_readback.json'))
    require(report_path==api.OUTPUT/(stage_name+'_readback.json') and not report_path.exists(),
            'C42 exclusive fixed readback path required')
    attempt=context.run/'saved_reader42_attempt.json'
    write_exclusive42(attempt,{'schema':'b22-tracking42-reader-attempt-v1','stage':stage_name,
        'release_identity':identity(release_path),'session_identity':identity(context.run/'session.json'),
        'reader_identity':identity(Path(__file__)),'retry_permitted':False})
    maths=math_factory(context.stage['reward_arm'],training_closure42(context))
    with geometry_counter42(maths.geometry,80000 if context.stage['mode']=='train' else 30600) as pure:
        details=(read_training42(context,maths) if context.stage['mode']=='train' else read_evaluation42(context,maths))
    require(pure['attempted']==pure['returned']==details['proof_pure_servo_intervals']
        and type(pure['returned']) is int and pure['returned']<=(80000 if context.stage['mode']=='train' else 30600),
        'C42 actual saved-reader pure geometry servo count differs')
    require(same_json42(api.validate_release42(Path(release_path)),context.release),'C42 post-read sources changed')
    require(api.directory_bytes42(context.run)<=context.session['quota_bytes'],'C42 reader output exceeded stage byte budget')
    assert_saved_only42()
    value={'schema':'b22-tracking42-readback-v1','status':'COMPLETE','execution_contract_id':CONTRACT,
        'stage':stage_name,'release_identity':identity(release_path),
        'session_identity':identity(context.run/'session.json'),'worker_receipt_identity':identity(context.run/'worker_receipt.json'),
        'reader_identity':identity(Path(__file__)),'actual_counts':context.actual,'actual_pure_servo_advances':pure['returned'],
        'pure_servo_counter':pure,'source_derivations42':maths.proofs,'reader_model_calls':0,'reader_physics_calls':0,
        'qualification_granted':False,**details,'artifacts':artifact_inputs42(context.run)}
    write_exclusive42(report_path,value)
    return value


def checked_saved_readback42(api,release_path,stage_name):
    """Only hashes and small summaries: comparison never replays saved geometry."""
    path=api.OUTPUT/(stage_name+'_readback.json');value=read_json(path);run=api.OUTPUT/stage_name
    require(value.get('schema')=='b22-tracking42-readback-v1' and value.get('status')=='COMPLETE'
        and value.get('stage')==stage_name and value.get('execution_contract_id')==CONTRACT
        and value.get('release_identity')==identity(release_path)
        and value.get('session_identity')==identity(run/'session.json')
        and value.get('worker_receipt_identity')==identity(run/'worker_receipt.json')
        and value.get('reader_identity')==identity(Path(__file__))
        and type(value.get('reader_model_calls')) is int and value['reader_model_calls']==0
        and type(value.get('reader_physics_calls')) is int and value['reader_physics_calls']==0,
        'C42 fixed complete saved readback required')
    for name,entry in value['artifacts'].items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and identity(run/relative)==entry,
                'C42 post-read raw artifact changed: '+name)
    stage=api.stage_by_name42(stage_name)
    api.checked_counts42(value['actual_counts'],stage,complete=True)
    require(type(value.get('actual_pure_servo_advances')) is int and 0<=value['actual_pure_servo_advances']<=(
        80000 if stage['mode']=='train' else 30600),'C42 saved readback pure geometry count differs')
    return value,path


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path)
    parser.add_argument('--stage',choices=tuple('train_'+arm for arm in ARMS))
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if not args.stage:
        require(args.release is None and args.report is None,'C44 a specific saved stage is required')
        print(json.dumps({'status':'PLAN_ONLY','reader_model_calls':0,'reader_physics_calls':0,'actual_pure_servo_advances':0}));return 0
    require(args.release is not None and args.report is not None and args.release.is_absolute() and args.report.is_absolute(),
            'C44 absolute release/report required')
    value=read_stage42(args.release,args.stage,args.report)
    print(json.dumps({'status':value['status'],'stage':value.get('stage'),'reader_physics_calls':0}));return 0


if __name__=='__main__':raise SystemExit(main())
