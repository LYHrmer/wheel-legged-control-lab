"""Pure synthetic pair/AST/process seams; no worker import, model or physics."""
import ast
import copy
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


runner=module('pair32_runner',ROOT/'scripts/run_b22_inference32.py')
reader=module('pair32_reader',ROOT/'scripts/read_b22_inference32.py')


def saved(mode,n=8):
    initial={'qpos':np.zeros(3),'qvel':np.zeros(3),'ctrl':np.zeros(16),
             'qacc_warmstart':np.zeros(3),'observation':np.zeros(99,dtype=np.float32),'time':np.asarray(0.)}
    states={key:np.repeat(value[None,...],n+1,axis=0) for key,value in initial.items()}
    states['time']=np.arange(n+1)*.01
    states['qpos'][:,0]=np.arange(n+1)*.001
    rows=[]; natives=[]
    for tick in range(n):
        gate=2<=tick<6
        predict=mode=='baseline' or gate
        action=[.125]*16 if predict else [0.]*16
        effective=[.125]*16 if gate else [0.]*16
        calc={'action_enabled':gate,'safe_torque_nm':[.25]*16,
              'wheel_integral_before_nm':[tick*.001]*4,'wheel_integral_after_nm':[(tick+1)*.001]*4,
              'wheel_common_reference_z_before_rad_s':tick*.01,
              'wheel_common_reference_z_after_rad_s':(tick+1)*.01,'applied_action':effective}
        adapter={'calculation':calc,'stop_latched_before':tick>5,'stop_latched_after':tick>=5}
        raw={'forward_velocity_mps':1.6 if gate else 0.,'yaw_rate_rps':0.}
        endpoint=lambda index:{key:states[key][index].tolist() for key in states}
        info={'applied_action':effective,'consumed_command':raw,'controller_record':adapter,'metrics':{'vx':.1},
              'reward_terms':{'action_rate':0.},'native_interval_summary':{'native_returns':5},
              'servo_receipt':{'tick':tick},'terminal_reason':'time_limit' if tick==n-1 else 'ongoing'}
        inference={'schema':'b22-gated-stateless-inference32-v1','actor':'B','checkpoint_sha256':runner.MODEL_SHA,
                   'action_gate_enabled':gate,'predict_called':predict,'policy_action_observed':predict,
                   'reason':'BASELINE_PREDICTED' if mode=='baseline' else 'PREDICTED' if gate else 'GATE_OFF',
                   'policy_output_kind':'SB3_clipped_deterministic_action' if predict else 'NOT_OBSERVED',
                   'raw_policy_action16':action if predict else None,'controller_input_action16':action}
        rows.append({'control_index':tick,'episode_tick':tick,'episode_index':0,'actor':'B',
                     'checkpoint_sha256':runner.MODEL_SHA,'input_observation99':states['observation'][tick].tolist(),
                     'raw_action16':action if predict else None,'action16':action,'policy_input_action':action,
                     'policy_predict_called':predict,'inference32':inference,'raw_command':raw,
                     'control_stages18':adapter,'native_actuator_traces':[{'applied_nm':[.25]*16} for _ in range(5)],
                     'pre_state':endpoint(tick),'post_state':endpoint(tick+1),'reward':.1,'info':info,
                     'terminated':False,'truncated':tick==n-1})
        natives.extend({'native_index':5*tick+j,'before':{'qpos':[tick*.001,0.,0.]},
                        'after':{'qpos':[(tick+1)*.001,0.,0.]},'contacts':[]} for j in range(5))
    data={'initial':initial,'states':states,'row_values':rows,'native_values':natives,'checked':[]}
    data['rows']=lambda:iter(data['row_values'])
    data['native']=lambda:iter(data['native_values'])
    data['validate']=lambda tick,row,five:data['checked'].append((tick,len(five)))
    return data


def test_same_full_states_and_all_native_rows_pass_with_only_gateoff_predictions_removed():
    a,b=saved('baseline'),saved('gated')
    result=reader.compare_numeric(a,b,8)
    assert result['control_predictions']=={'baseline':8,'gated':4}
    assert result['total_predict_calls_including_loader_probe']=={'baseline':9,'gated':5}
    assert result['control_predictions_saved']==result['gate_off_controls']==4
    assert a['checked']==b['checked']==[(tick,5) for tick in range(8)]


@pytest.mark.parametrize('fault',['qpos','qvel','PI','z','stop','effective','torque','native','command','reward','termination','observation'])
def test_any_tiny_physical_or_memory_difference_fails_without_tolerance(fault):
    a,b=saved('baseline'),saved('gated'); row=b['row_values'][3]
    if fault in ('qpos','qvel'):b['states'][fault][4,0]+=1e-15
    elif fault=='PI':row['control_stages18']['calculation']['wheel_integral_after_nm'][0]+=1e-15
    elif fault=='z':row['control_stages18']['calculation']['wheel_common_reference_z_after_rad_s']+=1e-15
    elif fault=='stop':row['control_stages18']['stop_latched_after']=True
    elif fault=='effective':row['info']['applied_action'][0]+=.01
    elif fault=='torque':row['control_stages18']['calculation']['safe_torque_nm'][0]+=1e-15
    elif fault=='native':b['native_values'][17]['after']['qpos'][1]+=1e-15
    elif fault=='command':row['raw_command']['yaw_rate_rps']=1e-15
    elif fault=='reward':row['reward']+=1e-15
    elif fault=='termination':row['terminated']=True
    else:row['input_observation99'][0]+=1e-15
    with pytest.raises(ValueError):reader.compare_numeric(a,b,8)


@pytest.mark.parametrize('fault',['missing_control','extra_control','missing_native','extra_native','short_states','wrong_dtype','initial'])
def test_complete_time_domain_and_initial_identity_are_required(fault):
    a,b=saved('baseline'),saved('gated')
    if fault=='missing_control':b['row_values'].pop()
    elif fault=='extra_control':b['row_values'].append(copy.deepcopy(b['row_values'][-1]))
    elif fault=='missing_native':b['native_values'].pop()
    elif fault=='extra_native':b['native_values'].append(copy.deepcopy(b['native_values'][-1]))
    elif fault=='short_states':b['states']['qpos']=b['states']['qpos'][:-1]
    elif fault=='wrong_dtype':b['states']['qvel']=b['states']['qvel'].astype(np.float32)
    else:b['initial']['qpos'][1]=1e-15
    with pytest.raises(ValueError):reader.compare_numeric(a,b,8)


@pytest.mark.parametrize('fault',['invent_output','called_off','not_called_on','wrong_gate','nonzero_zero','wrong_checkpoint'])
def test_inference_ledger_cannot_hide_calls_or_fabricate_unobserved_policy_output(fault):
    a,b=saved('baseline'),saved('gated')
    row=b['row_values'][0 if fault!='not_called_on' else 3]
    if fault=='invent_output':row['inference32']['raw_policy_action16']=[0.]*16
    elif fault=='called_off':row['policy_predict_called']=True
    elif fault=='not_called_on':row['inference32']['predict_called']=False
    elif fault=='wrong_gate':row['inference32']['action_gate_enabled']=True
    elif fault=='nonzero_zero':row['action16'][0]=.01
    else:row['inference32']['checkpoint_sha256']='f'*64
    with pytest.raises(ValueError):reader.compare_numeric(a,b,8)


def test_worker32_whole_module_is_exact_declared_patch_of_frozen31():
    runner.check_worker32(ROOT)
    expected=runner.derive_worker32((ROOT/runner.BASE_WORKER).read_text())
    actual=(ROOT/'scripts/run_b22_worker32.py').read_text()
    assert ast.dump(ast.parse(expected),include_attributes=False)==ast.dump(ast.parse(actual),include_attributes=False)
    assert 'env.controller.action_gate(env.decision.world_command)' in actual
    assert "runtime.control_step(env,action)" in actual


@pytest.mark.parametrize('fault',['base','unapproved_control_change'])
def test_undocumented_worker_changes_cannot_enter_release(tmp_path,fault):
    (tmp_path/'scripts').mkdir()
    original=(ROOT/runner.BASE_WORKER).read_text()
    worker=(ROOT/'scripts/run_b22_worker32.py').read_text()
    if fault=='base':original+='\n# changed base\n'
    else:worker=worker.replace('runtime.control_step(env,action)','runtime.control_step(env,action*0)')
    (tmp_path/runner.BASE_WORKER).write_text(original)
    (tmp_path/'scripts/run_b22_worker32.py').write_text(worker)
    with pytest.raises(ValueError):runner.check_worker32(tmp_path)


def test_default_plan_is_unvalidated_and_starts_nothing(tmp_path,capsys):
    output=tmp_path/'not_created'
    assert runner.main(['--output',str(output)])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['behavior_equivalence']=='NOT_VALIDATED' and plan['training_allowed'] is False
    assert plan['limits']['controls']==3200 and plan['limits']['normal_native']==16000 and plan['limits']['compiler_native']==4
    assert not output.exists()


def test_imports_are_cold_and_do_not_load_old_worker_or_heavy_reader():
    for filename in ('run_b22_inference32.py','read_b22_inference32.py'):
        code=(f'import importlib.util,sys;s=importlib.util.spec_from_file_location("cold32",{str(ROOT/"scripts"/filename)!r});'
              'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
              'assert not {"mujoco","torch","onnxruntime","stable_baselines3","reader26","run_gui26"}&sys.modules.keys()')
        process=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=5,check=False)
        assert process.returncode==0,process.stderr


@pytest.fixture
def lifecycle(tmp_path):
    state=SimpleNamespace(now=0.,cleaned=[],sampled=[])
    class Process:
        returncode=0
        pid=123456789
        def poll(self):return 0
    def clean(owner,process):
        assert all(signal.getsignal(sig)==signal.SIG_IGN for sig in (signal.SIGINT,signal.SIGTERM))
        state.cleaned.append(process)
        return {'remaining':{}}
    state.process=Process()
    state.host=SimpleNamespace(clean_descendants=clean)
    state.factory=lambda *args,**kwargs:state.process
    state.clock=lambda:state.now
    state.sample=lambda admission:state.sampled.append(admission)
    state.log=tmp_path/'process.log'
    return state


def invoke(state):
    return runner.supervise(['fake'],state.log,state.host,state.sample,300,900,
                            factory=state.factory,clock=state.clock,sleeper=lambda _:None)


def test_process_success_still_checks_ram_wall_and_owned_cleanup(lifecycle):
    assert invoke(lifecycle)==0 and lifecycle.sampled==[False] and lifecycle.cleaned==[lifecycle.process]


def test_exited_process_cannot_escape_wall_deadline(lifecycle):
    def factory(*args,**kwargs):lifecycle.now=301.;return lifecycle.process
    lifecycle.factory=factory
    with pytest.raises(ValueError,match='wall limit after exit'):invoke(lifecycle)
    assert lifecycle.cleaned==[lifecycle.process]


@pytest.mark.parametrize('sig',[signal.SIGINT,signal.SIGTERM])
def test_popen_signal_is_deferred_until_owned_handle_is_cleaned(lifecycle,sig):
    prior={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def factory(*args,**kwargs):signal.getsignal(sig)(sig,None);return lifecycle.process
    lifecycle.factory=factory
    with pytest.raises(ValueError,match='interrupted'):invoke(lifecycle)
    assert lifecycle.cleaned==[lifecycle.process]
    assert all(signal.getsignal(s)==old for s,old in prior.items())


def test_runtime_resource_exception_cleans_owned_child(lifecycle):
    lifecycle.process.poll=lambda:None
    def bad(_):raise ValueError('RAM below 4 GiB')
    lifecycle.sample=bad
    with pytest.raises(ValueError,match='RAM below'):invoke(lifecycle)
    assert lifecycle.cleaned==[lifecycle.process]


def test_cleanup_failure_cannot_be_accepted_as_process_success(lifecycle):
    lifecycle.host.clean_descendants=lambda *_:{'remaining':{'owned':[1]}}
    with pytest.raises(ValueError,match='cleanup incomplete'):invoke(lifecycle)


def test_model_counter_must_match_probe_plus_exact_control_calls():
    def result(predictions):
        values={'load':1,'torch_load':3,'predict':predictions+1,'save':0,'learn':0,'train':0,'forward':0,
                'evaluate_actions':0,'predict_values':0,'backward':0}
        counts={key:{'attempted':value,'returned':value} for key,value in values.items()}
        counts['predict'].update(rows_attempted=predictions+32,rows_returned=predictions+32)
        return {'model_calls':{'limits':{'predict':1601},'counts':counts,'actor_rows':predictions+32,'critic_rows':0},
                'policy_predictions':predictions}
    real=result(720);reader.model_counts(real,{'model_limits':{'predict':1601}},720)
    real['model_calls']['counts']['predict']['returned']=1601
    with pytest.raises(ValueError):reader.model_counts(real,{'model_limits':{'predict':1601}},720)


def test_frozen_physical_ledger_extraction_preserves_all_nonmodel_nodes():
    path=ROOT/'runtime/d1_b22/research/continuation26/root26/reader26.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='check_ledgers')
    starts=[ast.unparse(n.targets[0]) for n in node.body if isinstance(n,ast.Assign)]
    assert 'calls' in starts and '(edges, mailbox)' in starts


@pytest.fixture
def execution(tmp_path,monkeypatch):
    root=tmp_path/'repo'; (root/'results').mkdir(parents=True)
    host_file=root/'host.py';host_file.write_text('# fake host\n')
    keys=['load','torch_load','predict','save','learn','train','forward','evaluate_actions','predict_values','backward']
    config={'host':str(host_file),'source_hashes':{str(host_file):runner.identity(host_file)},
            'runtime_environment':{},'shared_session':{},'model_limit_keys':keys}
    state=SimpleNamespace(root=root,output=root/'results/b22_inference32_test',modes=[],
                          memory_values=[],mutation=None,cleanup_remaining={},fail_mode=None,short=None)
    def memory():
        value=state.memory_values.pop(0) if state.memory_values else 8<<30
        if isinstance(value,BaseException):raise value
        return value
    host=SimpleNamespace(subreaper=lambda:None,
                         clean_descendants=lambda *_:{'remaining':state.cleanup_remaining})
    monkeypatch.setattr(runner,'ROOT',root)
    monkeypatch.setattr(runner,'source_identity',lambda:{'git_commit':'a'*40,'authored':{}})
    monkeypatch.setattr(runner,'read_memory',memory)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=2<<30))
    monkeypatch.setattr(runner,'registered_config',lambda:(config,{'status':'READY'}))
    monkeypatch.setattr(runner,'validate_release',lambda path:runner.read_json(path))
    monkeypatch.setattr(runner,'load_module',lambda *_:host)
    monkeypatch.setattr(runner,'MANIFEST','manifest.json')
    (root/'manifest.json').write_text('{}\n')
    def supervised(argv,log,host,sample,limit,deadline):
        sample(False)
        if '--request' in argv:
            request=runner.read_json(argv[-1]);spec=request['spec'];mode=spec['inference_mode32']
            state.modes.append(mode)
            output=Path(spec['output_directory']);output.mkdir()
            go=runner.read_json(request['source_go_path'])
            session={**spec,'arm':mode,'checkpoint_sha256':runner.MODEL_SHA,'source_hashes':go['inputs']}
            runner.write_json(output/'session.json',session)
            count=state.short if state.short and mode=='baseline' else 1600
            result={'completed_controls':count,'inference_mode32':mode,
                    'python':{'control_attempted':count,'control_completed':count,'native_failed':0,
                              'native_attempted':5*count,'native_returned':5*count},
                    'C_final':{'construction_attempts':2,'construction_returns':2,
                               'control_attempts':5*count,'control_returns':5*count},
                    'terminated':count!=1600,'truncated':count==1600}
            worker={'execution_contract_id':'C32_B22_inference_pair_v1','arm':mode,
                    'session_identity':runner.identity(output/'session.json'),
                    'failure':None,'warnings':[],'result':result}
            if state.mutation:state.mutation(worker,session,output)
            runner.write_json(output/'worker_receipt.json',worker)
            return 1 if state.fail_mode==mode else 0
        destination=Path(argv[argv.index('--report')+1]);release_path=Path(argv[argv.index('--release')+1])
        runner.write_json(destination,{'status':'PASS','release_identity':runner.identity(release_path)})
        return 0
    monkeypatch.setattr(runner,'supervise',supervised)
    return state


def test_actual_execute_flow_is_baseline_then_gated_then_independent_readback(execution):
    report=runner.execute(execution.output)
    assert report['status']=='PASS' and execution.modes==['baseline','gated']
    assert report['actual_counts']=={'controls':3200,'normal_native':16000,'compiler_native':4}
    assert report['behavior_equivalence']=='PASS' and report['unknown_count_stages']==[]
    assert (execution.output/'release.json').is_file() and (execution.output/'comparison.json').is_file()


@pytest.mark.parametrize('value',[6*(1<<30)-1,OSError('cgroup unavailable')])
def test_below_six_gib_or_unreadable_memory_does_not_claim_attempt(execution,value):
    execution.memory_values=[value]
    with pytest.raises((ValueError,OSError)):runner.execute(execution.output)
    assert not execution.output.exists() and execution.modes==[]


def test_runtime_ram_read_error_stops_before_next_session(execution):
    execution.memory_values=[8<<30,8<<30,OSError('memory read failed')]
    report=runner.execute(execution.output)
    assert report['status']=='INCOMPLETE' and execution.modes==[]
    assert report['actual_counts'] is None and report['unknown_count_stages']==['baseline']


@pytest.mark.parametrize('value,passed', [(4*(1<<30)-1,False),(4<<30,True)])
def test_runtime_four_gib_boundary_after_six_gib_admission(execution,value,passed):
    execution.memory_values=[8<<30,8<<30,value]
    report=runner.execute(execution.output)
    assert runner.LIMITS['start_effective_bytes']==6<<30
    assert runner.LIMITS['reserve_effective_bytes']==4<<30
    assert (report['status']=='PASS') is passed
    if not passed:
        assert execution.modes==[] and report['behavior_equivalence']=='NOT_VALIDATED'
        assert report['unknown_count_stages']==['baseline']
        assert 'effective RAM below 4 GiB' in report['failure']


@pytest.mark.parametrize('fault',['checkpoint','mode','plan','worker_session'])
def test_foreign_worker_receipt_cannot_supply_known_physics_counters(execution,fault):
    def mutate(worker,session,output):
        if fault=='worker_session':worker['session_identity']={'sha256':'f'*64,'bytes':1}
        else:
            if fault=='checkpoint':session['checkpoint_sha256']='f'*64
            elif fault=='mode':session['inference_mode32']='gated'
            else:session['source_hashes'][str(execution.output/'release.json')]={'sha256':'f'*64,'bytes':1}
            (output/'session.json').write_text(json.dumps(session))
            worker['session_identity']=runner.identity(output/'session.json')
    execution.mutation=mutate
    report=runner.execute(execution.output)
    assert report['status']=='INCOMPLETE' and execution.modes==['baseline']
    assert report['actual_counts'] is None and report['known_actual_counts']=={'controls':0,'normal_native':0,'compiler_native':0}


def test_legitimate_short_task_is_retained_but_cannot_pass_equivalence(execution):
    execution.short=8
    report=runner.execute(execution.output)
    assert report['status']=='INCOMPLETE' and execution.modes==['baseline']
    assert report['actual_counts']=={'controls':8,'normal_native':40,'compiler_native':2}
    assert report['behavior_equivalence']=='NOT_VALIDATED'


def test_existing_attempt_never_repeats_either_session(execution):
    execution.output.mkdir()
    with pytest.raises(ValueError,match='new direct'):runner.execute(execution.output)
    assert execution.modes==[]


def test_final_cleanup_failure_revokes_success(execution):
    execution.cleanup_remaining={'owned':[123]}
    report=runner.execute(execution.output)
    assert report['status']=='INCOMPLETE' and report['behavior_equivalence']=='NOT_VALIDATED'
    assert report['actual_counts']=={'controls':3200,'normal_native':16000,'compiler_native':4}
