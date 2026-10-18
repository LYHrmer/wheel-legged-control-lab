"""C42 worker source/fake I/O gates. No Torch, simulator, env or NN loading."""
from __future__ import annotations

import ast
import copy
import importlib.util
import resource
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('worker42_pure',ROOT/'scripts/worker_b22_tracking_reward42.py')
w=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(w)


def quota(tmp_path,limit=8192,reserve=1024):
    return w.StageQuota42(tmp_path,limit,reserve=reserve)


@pytest.mark.parametrize('kind',['recipes','curriculum','checkpoint','heldout','train','metadata'])
def test_actual_modules_are_full_sha_pinned_inverse_ast_adaptations(kind):
    tree,proof=w.adapt_module42(kind)
    assert proof['inverse_ast_exact']
    compile(tree,str(w.WORKER),'exec')
    values=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    if kind=='heldout':
        assert 'd1-b22-tracking42-heldout-record-v1' in values
    if kind=='curriculum':
        assert '_quota42.open_text' in ast.unparse(tree)
    if kind=='train':
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert 'b22_curriculum42' in imports and 'b22_checkpoint42' in imports
        assert 'b22_metadata42' in imports and 'b22_recipes42' in imports


def test_worker_real_owner_inverse_lock_and_import_before_use():
    tree,proof=w.worker_ast42()
    assert proof['inverse_owner_ast_exact']
    compile(tree,str(w.WORKER),'exec')
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    owner=next(n for n in main.body if isinstance(n,ast.Try))
    lines=[ast.unparse(n) for n in owner.body]
    mj=lines.index('import mujoco')
    assert lines[mj+1]=='verify_mujoco42()'
    env=next(i for i,x in enumerate(lines) if x.startswith('from world_upright_course_11 import'))
    assert lines[env+1].startswith('WorldUprightCourseEnv = reward_env42')
    assert lines[env+2]=='install_servo_counter42(session)'
    text=ast.unparse(main)
    assert 'C22_fixed_B_eval_repair_v1' not in text
    assert 'C21 permits only fresh' not in text
    assert 'construction_limit=2' in text
    assert 'construction_returns' in text and '5 * actual' in text
    assert 'train = train42' in text and 'evaluate = evaluate42' in text
    assert 'install_controller18 = install_controller42' in text
    assert 'signal.signal(signal.SIGINT, signal_stop)' in text
    assert 'signal.signal(signal.SIGXCPU, signal_stop)' in text


def test_strict_clone_has_one_actual_shared_global_namespace():
    original=ModuleType('fake_original')
    exec(  # noqa: S102 -- synthetic functions
         "WORLD_UPRIGHT_REWARD_SCHEMA='old'\nFINAL_CHECKPOINT_SCHEMA='old-final'\n"
         "def helper(): return WORLD_UPRIGHT_REWARD_SCHEMA\n"
         "def load(): return helper(), FINAL_CHECKPOINT_SCHEMA\n",original.__dict__)
    module=w.strict_module42(original)
    assert module.load.__globals__ is module.__dict__
    assert module.helper.__globals__ is module.__dict__
    assert module.load()==(w.REWARD_SCHEMA,w.FINAL_SCHEMA)
    assert original.load()==('old','old-final')
    assert module.load.__code__ is original.load.__code__


@pytest.fixture
def production_installer42():
    """Execute only the actual pinned installer function, with synthetic chain types."""
    path=ROOT/w.SOURCES['controller'][0]
    assert w.identity(path)['sha256']==w.SOURCES['controller'][1]
    function=next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='install_controller18')
    cache={'currsize':1,'misses':1,'hits':4}
    class Adapter:
        def __init__(self,plant):
            self.plant=plant;self._preview=None;self.enable_stationary_action_probe=False
    class Loop:
        def __init__(self,plant,provider,controller,*,caps):
            self.plant=plant;self.provider=provider;self.controller=controller;self.caps=caps
            self._initialized=False;self._decision=None;self._receipt=None
    class Controller:
        def __init__(self,plant,*,mode,variant,enable_stationary_action_probe):
            self.plant=plant;self.mode=mode;self.variant=variant
            self.enable_stationary_action_probe=enable_stationary_action_probe
            cache['hits']+=1
    class Production:
        def __init__(self):
            self.plant=object();self.caps=object();self.mode='train';self.pure_test_components=False
            self._steps=None;self._episode_index=-1;self._active=False
            self.decision=None;self.last_transition=None
            self.controller=Adapter(self.plant)
            self.loop=Loop(self.plant,object(),self.controller,caps=self.caps)
    module=ModuleType('pinned_production_installer42')
    module.__dict__.update(WorldUprightCourseEnv=Production,WorldUprightLoop=Loop,
        FullDriveControllerAdapter=Adapter,FullDriveController18=Controller,
        VARIANTS=('baseline','yaw','ramp','combined'),
        _nominal_geometry=SimpleNamespace(cache_info=lambda:SimpleNamespace(**cache)))
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),module.__dict__)  # noqa: S102 -- exact function from SHA-pinned controller, no imports or physical objects
    return module,cache


def training_type42(module):
    return w.reward_env42(module.WorldUprightCourseEnv,{'alpha':8,'reward_arm':'deficit8'})


def test_real_old_installer_reproduces_reward_subclass_failure_then_private_mapping_passes(production_installer42):
    module,cache=production_installer42
    cls=training_type42(module);env=cls();old_loop=env.loop;old_controller=env.controller
    with pytest.raises(TypeError,match='C18 installer needs the exact production world-upright env'):
        module.install_controller18(env,'combined')
    install,proof=w.controller_installer42(module,cls)
    result=install(env,'combined')
    assert result is env.controller and result is not old_controller
    assert env.loop is not old_loop and env.loop.plant is env.plant
    assert env.loop.provider is old_loop.provider and env.loop.caps is env.caps
    assert cache=={'currsize':1,'misses':1,'hits':5}
    assert install.__code__ is module.install_controller18.__code__
    assert proof['globals_changed']==['WorldUprightCourseEnv']
    assert proof['function_code_unchanged'] and proof['old_module_globals_unchanged']
    assert module.install_controller18.__globals__ is module.__dict__
    assert module.WorldUprightCourseEnv is cls.__bases__[0]


@pytest.mark.parametrize('fault',['subclass_of_registered','unregistered_subclass','foreign_base'])
def test_controller_mapping_does_not_accept_arbitrary_subclasses(production_installer42,fault):
    module,_=production_installer42
    registered=training_type42(module)
    if fault=='subclass_of_registered':cls=type('Other',(registered,),{})
    elif fault=='unregistered_subclass':cls=type('Other',(module.WorldUprightCourseEnv,),{})
    else:cls=w.reward_env42(type('Foreign',(),{}),{'alpha':1,'reward_arm':'control'})
    with pytest.raises(ValueError,match='exact registered training env'):
        w.controller_installer42(module,cls)


@pytest.mark.parametrize('fault',['pure_test','loop_type','controller_type','already_reset','active',
                                 'prepared_decision','controller_preview','cold_cache','foreign_plant'])
def test_real_installer_keeps_every_existing_source_chain_guard(production_installer42,fault):
    module,cache=production_installer42;cls=training_type42(module);env=cls()
    if fault=='pure_test':env.pure_test_components=True
    elif fault=='loop_type':
        env.loop=type('LoopChild',(module.WorldUprightLoop,),{})(env.plant,env.loop.provider,env.controller,caps=env.caps)
    elif fault=='controller_type':
        env.controller=type('ControllerChild',(module.FullDriveControllerAdapter,),{})(env.plant)
        env.loop.controller=env.controller
    elif fault=='already_reset':env._steps=0
    elif fault=='active':env._active=True
    elif fault=='prepared_decision':env.decision=object()
    elif fault=='controller_preview':env.controller._preview=object()
    elif fault=='cold_cache':cache['currsize']=0
    else:env.loop.plant=object()
    install,_=w.controller_installer42(module,cls)
    with pytest.raises((TypeError,RuntimeError)):install(env,'combined')
    assert cache['hits']==4


def test_evaluation_uses_original_installer_function_without_mapping(production_installer42):
    module,_=production_installer42
    install,proof=w.controller_installer42(module,module.WorldUprightCourseEnv)
    assert install is module.install_controller18 and proof['globals_changed']==[]
    assert not proof['registered_training_env']
    env=module.WorldUprightCourseEnv();install(env,'combined')
    assert type(env.controller) is module.FullDriveController18


def test_actual_worker_installer_proxy_calls_fixed_body_and_archives_mapping(production_installer42,tmp_path,monkeypatch):
    module,_=production_installer42;env=training_type42(module)()
    monkeypatch.setattr(w,'controller_module42',lambda:module)
    monkeypatch.setattr(w,'_quota42',quota(tmp_path))
    controller=w.install_controller42(env,'combined')
    assert controller is env.controller
    saved=w.read_json(tmp_path/'controller_installer42_proof.json')
    assert saved['function_code_unchanged'] and saved['installation_completed']
    assert saved['source_sha256']==w.SOURCES['controller'][1]
    assert saved['globals_changed']==['WorldUprightCourseEnv']


@pytest.mark.parametrize('alpha',[1,8])
def test_fake_real_reward_env_hook_once_no_actor_or_terminal_mutation(alpha):
    terms={'tracking_vx':1.99,'tracking_vy':1.,'tracking_yaw':.98,'height':-.1,
           'attitude':-.02,'actual_torque':-.03,'action_change':-.04,'leg_speed':-.05,'termination':-10.}
    actor=object();info={'reward_terms':terms};seen=[]
    class Base:
        reward_schema='old'
        def _reward(self,*_):
            seen.append('reward')
            return terms
        def step(self,action):
            assert action is actor
            value=self._reward(None,None,True)
            info['reward_terms']=value
            return actor,sum(value.values()),True,False,info
    cls=w.reward_env42(Base,{'alpha':alpha,'reward_arm':'control' if alpha==1 else 'deficit8'})
    env=cls();snapshot=copy.copy(terms)
    observation,reward,terminated,truncated,result=env.step(actor)
    assert seen==['reward'] and observation is actor and terminated and not truncated
    assert result is not info and result['original_reward_terms42']==snapshot and terms==snapshot
    assert result['training_reward42']=={'schema':w.REWARD_SCHEMA,'alpha':alpha,
                                        'reward_arm':'control' if alpha==1 else 'deficit8'}
    for key, value in terms.items():
        if key not in ('tracking_vx','tracking_yaw'):
            assert result['reward_terms'][key]==value
    assert reward==sum(result['reward_terms'].values())
    if alpha==1:
        assert result['reward_terms']==snapshot
    else:
        assert result['reward_terms']['tracking_vx']==2+8*(terms['tracking_vx']-2)


def test_binary_stream_limit_is_before_write_and_partial_is_kept(tmp_path):
    q=quota(tmp_path,limit=110,reserve=10)
    path=tmp_path/'partial'
    with pytest.raises(w.QuotaExceeded42):
        q.stage(path,lambda stream:(stream.write(b'a'*80),stream.write(b'b'*30)))
    assert path.read_bytes()==b'a'*80
    assert q.payload==80 and q.exhausted
    q.receipt_json(tmp_path/'r',{})
    assert q.initial+q.payload+q.reserved<=q.limit


def test_seek_overwrites_do_not_double_count_file_growth(tmp_path):
    q=quota(tmp_path)
    def save(stream):
        stream.write(b'a'*100);stream.seek(0);stream.write(b'b'*20)
    q.stage(tmp_path/'x',save)
    assert q.payload==100 and (tmp_path/'x').stat().st_size==100


def test_utf8_event_stream_is_charged_and_closed_on_real_file(tmp_path):
    q=quota(tmp_path)
    stream=q.open_text(tmp_path/'events')
    assert stream.write('中文\n')==3
    stream.flush()
    assert q.payload==len('中文\n'.encode())
    stream.close()
    assert stream.closed


def test_actual_atomic_writer_rejects_before_payload_growth_and_latches(tmp_path):
    path=ROOT/'runtime/d1_b22/research/continuation13/archive13/atomic_archive_13.py'
    tree=ast.parse(path.read_text())
    stop=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.ImportFrom)
              and n.module=='course_impl08.course_native_guard_08')
    module=ModuleType('quota42_actual_archive')
    exec(compile(ast.Module(body=tree.body[:stop],type_ignores=[]),str(path),'exec'),  # noqa: S102 -- pinned ArchiveWriter prefix
         module.__dict__)
    q=quota(tmp_path,limit=900,reserve=100)
    cls=w.quota_writer42(module.ArchiveWriter,q,lambda v:v)
    writer=cls()
    with pytest.raises(w.QuotaExceeded42):
        writer.commit_json(tmp_path/'big.json',{'x':'a'*1000})
    assert writer.failed and not (tmp_path/'big.json.manifest.json').exists()
    assert all(p.stat().st_size<=800 for p in tmp_path.iterdir() if p.is_file())


def test_initial_and_global_existing_files_are_counted(tmp_path):
    stage=tmp_path/'stage';stage.mkdir()
    (stage/'session').write_bytes(b'a'*100)
    q=w.StageQuota42(stage,1000,execution_output=tmp_path,reserve=100)
    assert q.initial==100
    (tmp_path/'external').write_bytes(b'x'*1000)
    q.global_stop=500
    with pytest.raises(w.QuotaExceeded42):
        q.stage(stage/'next',lambda stream:stream.write(b'x'))
    assert not (stage/'next').exists()


def test_checkpoint_kernel_limit_and_reservation_restore_after_error(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'EXTERNAL_FILE_LIMIT',64)
    q=quota(tmp_path)
    previous=resource.getrlimit(resource.RLIMIT_FSIZE)
    handler=signal.getsignal(signal.SIGXFSZ)
    with pytest.raises((OSError,w.QuotaExceeded42)), q.external_files(1), (tmp_path/'checkpoint').open('wb') as stream:
        stream.write(b'x'*65)
        stream.flush()
    assert (tmp_path/'checkpoint').stat().st_size<=64
    assert q.payload==64
    assert resource.getrlimit(resource.RLIMIT_FSIZE)==previous
    assert signal.getsignal(signal.SIGXFSZ)==handler


def test_checkpoint_reservation_cannot_start_after_budget_is_used(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'EXTERNAL_FILE_LIMIT',256)
    q=quota(tmp_path,limit=1000,reserve=100)
    called=False
    with pytest.raises(w.QuotaExceeded42), q.external_files(4):
        called=True
    assert not called and q.exhausted


def test_real_checkpoint_sources_have_bounded_file_targets():
    # Final checkpoint has one model plus three fixed _write_exclusive calls;
    # failure has one model plus one fixed metadata. Imports cause no writes.
    final=ast.parse((ROOT/w.SOURCES['checkpoint'][0]).read_text())
    func=next(n for n in final.body if isinstance(n,ast.FunctionDef) and n.name=='save_final_and_verify')
    direct=[n for n in ast.walk(func) if isinstance(n,ast.Call)]
    assert sum(isinstance(n.func,ast.Name) and n.func.id=='_write_exclusive' for n in direct)==3
    assert sum(isinstance(n.func,ast.Attribute) and n.func.attr=='save' for n in direct)==1
    strict=ast.parse((ROOT/w.SOURCES['strict'][0]).read_text())
    func=next(n for n in strict.body if isinstance(n,ast.FunctionDef) and n.name=='save_failure_checkpoint')
    direct=[n for n in ast.walk(func) if isinstance(n,ast.Call)]
    assert sum(isinstance(n.func,ast.Name) and n.func.id=='_write_exclusive' for n in direct)==1
    assert sum(isinstance(n.func,ast.Attribute) and n.func.attr=='save' for n in direct)==1


def test_nominal_servo_counter_calls_original_once_filters_live_and_restores():
    returned=object()
    class Servo:
        def advance(self,*_):
            return returned
    original=Servo.advance
    counter=w.ServoCounter42(Servo,3).install()
    namespace={'__name__':'b22_curriculum42','servo':Servo()}
    exec('def _nominal_path(): return servo.advance(0,None)',namespace)  # noqa: S102 -- synthetic caller identity
    assert namespace['_nominal_path']() is returned
    assert Servo().advance(0,None) is returned
    assert counter.returned==counter.attempted==1 and counter.other_returned==1
    assert counter.by_caller=={'b22_curriculum42:_nominal_path':1}
    counter.close()
    assert Servo.advance is original


def test_failed_pure_call_retains_attempt_but_not_false_actual_count(monkeypatch,tmp_path):
    class Servo:
        def advance(self,*_):
            raise ValueError('actual advance failed')
    counter=w.ServoCounter42(Servo,3).install()
    namespace={'__name__':'geometry21','servo':Servo()}
    exec('def nominal_path21(): return servo.advance(0,None)',namespace)  # noqa: S102 -- synthetic caller identity
    try:
        with pytest.raises(ValueError):
            namespace['nominal_path21']()
        monkeypatch.setattr(w,'_servo_counter42',counter,raising=False)
        value=receipt(complete=False)
        out=w.final_receipt42(value,{'mode':'train','controls':32768,'alpha':8,
            'reward_arm':'deficit8','pure_servo_advance_max':160000},quota(tmp_path))
        assert out['actual_pure_servo_advances'] is None and out['known_pure_servo_returns']==0
        assert out['pure_servo42']['attempted']==1
    finally:
        counter.close()


def receipt(complete=True,n=32768):
    return {'execution_complete':complete,'stop_latched':False,'failure':None,'cleanup_errors':[],
            'python':{'control_completed':n,'native_returned':n*5},
            'C_final':{'construction_returns':2}}


@pytest.mark.parametrize('mode,n',[('train',32768),('eval',12000)])
def test_completed_counts_use_actual_and_preserve_short_task_eval(monkeypatch,tmp_path,mode,n):
    counter=SimpleNamespace(returned=2000,attempted=2000,report=lambda:{'attempted':2000,'returned':2000})
    monkeypatch.setattr(w,'_servo_counter42',counter,raising=False)
    s={'mode':mode,'controls':32768 if mode=='train' else 30600,'alpha':1 if mode=='train' else None,
       'reward_arm':'control' if mode=='train' else None,'pure_servo_advance_max':160000}
    out=w.final_receipt42(receipt(n=n),s,quota(tmp_path))
    assert out['status']=='COMPLETE' and out['actual_counts']['controls']==n
    assert out['actual_counts']['normal_native']==5*n and out['actual_pure_servo_advances']==2000


def test_bad_success_envelope_becomes_incomplete_not_lost_receipt(monkeypatch,tmp_path):
    monkeypatch.delattr(w,'_servo_counter42',raising=False)
    s={'mode':'train','controls':32768,'alpha':1,'reward_arm':'control','pure_servo_advance_max':160000}
    out=w.final_receipt42(receipt(n=12),s,quota(tmp_path))
    assert out['status']=='INCOMPLETE' and out['actual_counts'] is None and out['failure']


def test_old_stage2_warmstart_seed_and_optimizer_hash_order_source_only():
    tree=ast.parse((ROOT/w.SOURCES['learning'][0]).read_text())
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_warm_start20')
    text=ast.unparse(func)
    assert 'stage2 Adam moments differ' in text and '1280 + parent_actual' in text
    assert text.index('model.set_random_seed(seed)')<text.index('model.num_timesteps = 0')
    assert 'setting stage seed changed parent parameters or Adam moments' in text
    assert 'stage initial hashes are not bitwise parent hashes' in text
    train,_=w.adapt_module42('train')
    text=ast.unparse(train)
    assert 'reset_num_timesteps=True' in text and "['stage'] - 1" in text
    assert text.index('model.learn(')<text.index('save_final_and_verify(')


def test_cold_worker_import_does_not_import_frameworks():
    code="""
import importlib.abc,importlib.util,sys
blocked={'torch','numpy','mujoco','gymnasium','stable_baselines3','cv2','onnxruntime'}
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in blocked: raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
s=importlib.util.spec_from_file_location('worker42_cold',sys.argv[1])
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert not blocked.intersection(sys.modules)
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(w.WORKER)],
                          capture_output=True,text=True,timeout=15,check=False)
    assert result.returncode==0,result.stderr


def test_failed_writer_rescue_uses_reserved_bounded_transaction(tmp_path):
    path=ROOT/'runtime/d1_b22/research/continuation13/archive13/atomic_archive_13.py'
    tree=ast.parse(path.read_text())
    stop=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.ImportFrom)
              and n.module=='course_impl08.course_native_guard_08')
    module=ModuleType('actual42_archive_rescue')
    exec(compile(ast.Module(body=tree.body[:stop],type_ignores=[]),str(path),'exec'),module.__dict__)  # noqa: S102 -- pinned stdlib writer prefix
    q=quota(tmp_path,limit=1500,reserve=400)
    writer=w.quota_writer42(module.ArchiveWriter,q,lambda value:value)()
    with pytest.raises(w.QuotaExceeded42):
        writer.commit_json(tmp_path/'payload',{'x':'a'*1600})
    result=writer.commit_failure_receipt(tmp_path/'failure',{'failure':{'type':'quota'}})
    assert (tmp_path/result['manifest']).is_file()
    assert q.reserved>0 and q.initial+q.payload+q.reserved<=q.limit
    assert writer.failed and writer.failure_receipt_attempted


def test_tree_snapshot_skips_only_disappeared_file_but_rejects_symlink(tmp_path,monkeypatch):
    file=tmp_path/'gone';file.write_bytes(b'x')
    original=Path.lstat
    def disappearing(path,*args,**kwargs):
        if path==file:
            raise FileNotFoundError('renamed')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',disappearing)
    assert w.tree_bytes42(tmp_path)==0
    monkeypatch.setattr(Path,'lstat',original)
    (tmp_path/'link').symlink_to(file)
    with pytest.raises(ValueError):
        w.tree_bytes42(tmp_path)


def test_eval_collects_both_arms_all_six_after_legitimate_failed_floors(tmp_path,monkeypatch):
    calls=[];runtime=SimpleNamespace(control_completed=0)
    recipes=SimpleNamespace(ORDER=tuple('case'+str(i) for i in range(6)),
        floor_schedule=lambda seed:SimpleNamespace(case_id='floor',control_cap=600,seed=seed),
        make_schedule=lambda case:SimpleNamespace(case_id=case,control_cap=1800 if case=='case5' else 1600,seed=1))
    def record(runtime,guard,env,policy,schedule,actor,folder,*,pair_initial,checkpoint_sha256,**kwargs):
        floor=schedule.case_id=='floor'
        n=12 if floor else schedule.control_cap
        if actor=='zero':assert policy is None and checkpoint_sha256 is None
        calls.append((schedule.case_id,actor,pair_initial))
        runtime.control_completed+=n
        row={'case_id':schedule.case_id,'actor':actor,'seed':schedule.seed,'completed_controls':n,
             'record_valid':True,'terminated':floor,'truncated':not floor,
             'stop_reason':'fall_or_low_clearance' if floor else 'time_limit',
             'policy_predictions':0 if actor=='zero' else n,'checkpoint_sha256':checkpoint_sha256}
        return row,{'first':schedule.case_id}
    heldout=SimpleNamespace(record_case=record)
    strict=SimpleNamespace(load_and_verify_final=lambda folder,manifest,**kwargs:(object(),{'loaded':folder}))
    monkeypatch.setattr(w,'runtime_modules42',lambda *a:(recipes,strict,None,heldout))
    monkeypatch.setattr(w,'_quota42',object())
    (tmp_path/'construction_receipt.json').write_text('{"geometry_binding":{},"compiled_geometry":{}}')
    seen=[]
    writer=SimpleNamespace(commit_json=lambda path,value:seen.append((path.name,value)))
    session={'eval_manifests':{a:{'folder':'/new/'+a,'files':{'final_model.zip':{'sha256':'a'*64}}}
                               for a in ('control','deficit8')},
             'source_hashes':{},'evaluation_split':'tracking42_original_six','floor_seed':201090}
    result=w.evaluate42(session,runtime,None,object(),tmp_path,writer,SimpleNamespace(phase=None))
    assert len(result['cases'])==18 and result['enabled_arms']==['control','deficit8']
    assert len(calls)==20 and runtime.control_completed==29424
    assert all(result['floors'][a]['receipt']['terminated'] for a in result['floors'])
    assert [actor for case,actor,_ in calls[2:5]]==['zero','control','deficit8']
    assert calls[2][2] is None and calls[3][2]==calls[4][2]=={'first':'case0'}
    assert result['evaluation_gain']==1 and result['evaluation_reward_unchanged']
    identity=next(value for name,value in seen if name=='eval_identity.json')
    assert identity['execution_contract_id']==w.CONTRACT and identity['execution_stage']==42


def test_bootstrap_repair_precedes_guard_without_resetting_native_counters():
    tree=ast.parse(w.WORKER.read_text())
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bootstrap_components42')
    source=ast.unparse(func)
    assert source.index('import_repair42(')<source.index('binding.EngineGuard(')<source.index('guard.snapshot(')
    assert 'guard.arm(' not in source and 'guard.set_phase(' not in source
    assert 'guard.set_target(' not in source


@pytest.mark.parametrize('bad_bindings',[False,True])
@pytest.mark.parametrize('attempt_constructor',[None,'environment','ppo','base_algorithm','nn_module'])
def test_bootstrap_runs_only_definition_graph_inside_zero_model_counter(tmp_path,monkeypatch,bad_bindings,attempt_constructor):
    import json
    state=SimpleNamespace(active=False,closed=False)
    native=dict.fromkeys(w.ZERO_NATIVE42,0);native['original_step']=0
    library='/fixed/epa.so';elf='/fixed/extension.so'
    binding={'passed':True,'dso_path':library,'dso_sha256':'a'*64,'counters':native,
             'elf_hashes':{elf:'b'*64},'jump_slots':[{'passed':True,'target':5,'expected_wrapper':5,
                                                   'elf':elf,'elf_sha256':'b'*64} for _ in range(4)]}
    def binding_proof():
        native['original_step']=12345  # Actual binding lazily resolves RTLD_NEXT without stepping.
        return binding
    guard=SimpleNamespace(snapshot=lambda:dict(native),binding_proof=binding_proof)
    cold={'hits':0,'misses':0,'maxsize':1,'currsize':0}
    geometry=SimpleNamespace(cache_info=lambda:SimpleNamespace(_asdict=lambda:dict(cold)))
    class Base:
        def __init__(self):
            pytest.fail('bootstrap constructed an environment')
    targets={'environment':Base,**{name:type(name,(),{}) for name in ('ppo','base_algorithm','nn_module')}}
    originals={name:kind.__dict__.get('__init__') for name,kind in targets.items()}
    monkeypatch.setattr(w,'constructor_targets42',lambda base:targets)
    controller=ModuleType('fake_bootstrap_controller42');controller.WorldUprightCourseEnv=Base
    exec('def install_controller18(env,variant): raise AssertionError("bootstrap called installer")',controller.__dict__)  # noqa: S102 -- synthetic function must never be called
    monkeypatch.setattr(w,'controller_module42',lambda:controller)
    class Counter:
        def __init__(self,training,limits):
            assert training is False and all(v==0 for v in limits.values())
            self.counts={}
        def __enter__(self):
            state.active=True;return self
        def __exit__(self,*args):
            state.active=False
        def summary(self):
            return {'counts':self.counts}
    monkeypatch.setattr(w,'bootstrap_components42',lambda *a:(guard,dict(native),Base,geometry,Counter))
    servo=SimpleNamespace(attempted=0,returned=0,other_returned=0,
        report=lambda:{'attempted':0,'returned':0},close=lambda:setattr(state,'closed',True))
    monkeypatch.setattr(w,'install_servo_counter42',
                        lambda s:setattr(w,'_servo_counter42',servo))
    def modules(env_type,session,q):
        assert state.active and session['alpha']==1
        if attempt_constructor:
            try:targets[attempt_constructor]()
            except RuntimeError:pass  # A swallowed constructor exception must still invalidate bootstrap.
        curriculum=ModuleType('b22_curriculum42');curriculum.__file__=str(w.WORKER)
        curriculum.WorldUprightCourseEnv=env_type;curriculum.WORLD_UPRIGHT_REWARD_SCHEMA=w.REWARD_SCHEMA
        monkeypatch.setitem(sys.modules,'b22_curriculum42',curriculum)
        original=ModuleType('fake_strict_bootstrap')
        exec('def load_and_verify_final(): return None',original.__dict__)  # noqa: S102 -- synthetic function only
        strict=w.strict_module42(original)
        if bad_bindings:strict.WORLD_UPRIGHT_REWARD_SCHEMA='wrong'
        train=ModuleType('b22_train42')
        exec('def train(): return None',train.__dict__)  # noqa: S102 -- synthetic function only
        return (SimpleNamespace(T={'ppo_seed_by_stage':[201001,201011]}),strict,train,
                SimpleNamespace(ACTORS=('zero','control','deficit8')))
    monkeypatch.setattr(w,'runtime_modules42',modules)
    session={'host_started_monotonic':__import__('time').monotonic(),'soft_s':10,
             'model_limits':{'load':0,'predict':0},'library':library,
             'source_hashes':{library:{'sha256':'a'*64},elf:{'sha256':'b'*64}}}
    (tmp_path/'session.json').write_text(json.dumps(session))
    q=quota(tmp_path,limit=64<<20,reserve=1<<20)
    monkeypatch.setattr(w,'_quota42',q)
    code=w.bootstrap42(session,tmp_path,q)
    receipt=json.loads((tmp_path/'worker_receipt.json').read_text())
    assert state.closed and not state.active
    assert {name:kind.__dict__.get('__init__') for name,kind in targets.items()}==originals
    if bad_bindings or attempt_constructor:
        assert code==1 and receipt['status']=='INCOMPLETE' and receipt['actual_counts'] is None
        if attempt_constructor:assert receipt['construction_guard42']['attempted'][attempt_constructor]==1
    else:
        assert code==0 and receipt['bootstrap_verified42']
        assert receipt['actual_counts']=={'controls':0,'normal_native':0,'compiler_native':0}
        proof=json.loads((tmp_path/'bootstrap42_proof.json').read_text())
        assert proof['state_after_dependency_imports']['original_step']==0
        assert proof['state_before_adapters']['original_step']==12345
        assert all(proof['state_after_dependency_imports'][k]==0 for k in w.ZERO_NATIVE42)
        assert proof['state_before_adapters']==proof['state_after']
        assert proof['environment_constructions']==proof['policy_construction_attempts']==0
        assert not any(proof['construction_guard42']['attempted'].values())
        assert proof['model_calls']=={'counts':{}}
