"""C38 saved metadata/fake process tests; no model, physics or native bootstrap."""
from __future__ import annotations

import copy
import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value
r = load('test_numpy38_runner',ROOT/'scripts/run_b22_numpy38.py')
d = load('test_numpy38_reader',ROOT/'scripts/read_b22_numpy38.py')


def test_plan_cold_fixed_budget_and_two_actual_profiles(capsys,monkeypatch):
    monkeypatch.setattr(r,'execute',lambda *args:pytest.fail('executed'))
    assert r.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['stage_order'] == ['bootstrap','flat_1p6_B','yaw_1p2_B']
    assert value['limits']['controls'] == 3200 and value['limits']['normal_native'] == 16000
    assert value['limits']['compiler_native'] == 4 and value['limits']['numpy_predict_rows'] == 3264
    assert value['limits']['numpy_predict_calls'] == 3202 and value['limits']['ppo_load'] == value['limits']['torch_load'] == 0
    assert value['relative_sse_and_torque_upper_factor'] == 1.001 and not value['qualification_granted']
    assert [row['seed'] for row in value['stages']] == [181202,181203]
    assert all(row['actor'] == 'B' and row['gain38'] == .5 for row in value['stages'])


def ledger(count=1600):
    return {'schema':'b22-numpy38-model-api-ledger-v1','backend':'NumPy','limits':dict(r.backend.API_LIMITS),
        'forbidden_modules_absent':True,'actor_rows':count+32,'critic_rows':0,'numpy_actor_bundle_loads':1,
        'ppo_load_calls':0,'torch_load_calls':0,'counts':{'predict':{
            'attempted':count+1,'returned':count+1,'rows_attempted':count+32,'rows_returned':count+32,
            'phases':{'probe':{'attempted':1,'returned':1,'rows':32},
                      'control':{'attempted':count,'returned':count,'rows':count}}}}}


@pytest.mark.parametrize('fault',[None,'old_schema','torch','skip_closed_gate','extra_probe','unreturned',
                                 'typed_phase','typed_limit','no_forbidden_guard'])
def test_new_numpy_ledger_never_uses_old_modelcounter_identity(fault):
    calls = ledger()
    if fault == 'old_schema':calls['schema'] = 'old-ppo-ledger'
    elif fault == 'torch':calls['torch_load_calls'] = 3
    elif fault == 'skip_closed_gate':calls['counts']['predict']['returned'] -= 880
    elif fault == 'extra_probe':calls['counts']['predict']['phases']['probe']['rows'] = 64
    elif fault == 'unreturned':calls['counts']['predict']['returned'] -= 1
    elif fault == 'typed_phase':calls['counts']['predict']['phases']['probe']['returned'] = True
    elif fault == 'typed_limit':calls['limits']['load'] = False
    elif fault == 'no_forbidden_guard':calls['forbidden_modules_absent'] = False
    if fault:
        with pytest.raises(ValueError):d.verify_numpy_ledger({'model_calls':calls},{'model_limits':r.backend.API_LIMITS},1600)
    else:d.verify_numpy_ledger({'model_calls':calls},{'model_limits':r.backend.API_LIMITS},1600)


def bootstrap():
    worker = load('test_bootstrap_schema38',ROOT/'scripts/run_b22_numpy_worker38.py')
    sources = {'/lib':{'sha256':'a'*64},'/numpy':{'sha256':'b'*64},'/elf':{'sha256':'c'*64}}
    state = dict.fromkeys(worker.ZERO_FIELDS,0)
    slots = [{'elf':'/elf','symbol':symbol,'relocation_offset':i,'passed':True,
              'target':17,'expected_wrapper':17,'elf_sha256':'c'*64}
             for i,symbol in enumerate(['mjc_ccd','mj_step','mj_step','mj_step'])]
    proof = {'passed':True,'dso_path':'/lib','dso_sha256':'a'*64,'counters':dict(state),
             'jump_slots':slots,'elf_hashes':{'/elf':'c'*64}}
    value = {'schema':'b22-numpy38-bootstrap-v1','status':'COMPLETE','execution_contract_id':r.CONTRACT,
        'session_identity':{'bytes':17,'sha256':'f'*64},
        'qualification_granted':False,'forbidden_modules_absent':True,'forbidden_prefixes':sorted(r.backend.FORBIDDEN),
        'environment_constructions':0,'numpy_actor_bundle_loads':0,'actual_counts':dict(r.counted({})),
        'actual_nn_counts':dict.fromkeys(('predict_calls','predict_rows','ppo_load','torch_load'),0),
        'nominal_cache':{'hits':0,'misses':0,'maxsize':1,'currsize':0},'state_before':dict(state),'state_after':dict(state),
        'binding_proof':proof,'module_origins':{'numpy':'/numpy'}}
    return value,{'source_hashes':sources,'library':'/lib'},{'bytes':17,'sha256':'f'*64}


@pytest.mark.parametrize('fault',[None,'native','nn','env','cache','arm','got','duplicate_got','foreign_elf','framework','qual','foreign_session'])
def test_saved_zero_bootstrap_proves_no_environment_nn_or_native(fault):
    value,session,expected = bootstrap()
    if fault == 'native':value['actual_counts']['compiler_native'] = 2
    elif fault == 'nn':value['actual_nn_counts']['predict_calls'] = 1
    elif fault == 'env':value['environment_constructions'] = 1
    elif fault == 'cache':value['nominal_cache']['misses'] = False
    elif fault == 'arm':value['state_after']['armed'] = True
    elif fault == 'got':value['binding_proof']['jump_slots'][0]['target'] = 19
    elif fault == 'duplicate_got':value['binding_proof']['jump_slots'][0] = value['binding_proof']['jump_slots'][1]
    elif fault == 'foreign_elf':value['binding_proof']['elf_hashes']['/elf'] = 'e'*64
    elif fault == 'framework':value['module_origins']['torch'] = '/numpy'
    elif fault == 'qual':value['qualification_granted'] = True
    elif fault == 'foreign_session':value['session_identity']['sha256'] = 'e'*64
    if fault:
        with pytest.raises(ValueError):d.verify_bootstrap(value,session,expected)
    else:assert d.verify_bootstrap(value,session,expected)


@pytest.fixture
def cold_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    output = tmp_path/'results/new38'
    output.mkdir(parents=True)
    monkeypatch.setattr(r,'OUTPUT',output)
    cfg = {'shared_session':{'library':'/lib','binding_module':'/binding','compiler_native_limit':2},
           'runtime_environment':{'LD_BIND_NOW':'1'}}
    path = tmp_path/r.c.MANIFEST
    path.parent.mkdir(parents=True,exist_ok=True)
    r.c.write_json(path,cfg)
    release = {'source':{'new38':True},'inputs':{'/lib':{'sha256':'a'*64}}}
    release_path = output/'release.json'
    r.c.write_json(release_path,release)
    monkeypatch.setattr(r,'validate_release',lambda path:release)
    argv = r.bootstrap_request(output,release,release_path,'e'*48)
    session_path = output/'bootstrap/session.json'
    session = r.c.read_json(session_path)
    session['parent_pid'] = r.os.getppid()
    r.atomic_json(session_path,session)
    journal = {'status':'RUNNING','active_stage':'bootstrap','supervisor_pid':r.os.getppid(),
        'release_identity':r.c.identity(release_path),'source':release['source'],'nonce':'e'*48,'active_command':argv}
    r.atomic_json(output/'report.json',journal)
    return session_path,session,journal


@pytest.mark.parametrize('fault',[None,'pid','nonce','argv','source','not_running','compiler_reservation'])
def test_cold_bootstrap_needs_live_new_parent_before_any_loader(cold_launch,fault):
    path,session,journal = cold_launch
    if fault == 'pid':journal['supervisor_pid'] += 1
    elif fault == 'nonce':session['nonce'] = 'old'
    elif fault == 'argv':session['argv'].append('--skip')
    elif fault == 'source':session['source_hashes']['/lib'] = {'sha256':'f'*64}
    elif fault == 'not_running':journal['status'] = 'COMPLETE'
    elif fault == 'compiler_reservation':session['compiler_native_limit'] = 2
    r.atomic_json(path,session)
    r.atomic_json(r.OUTPUT/'report.json',journal)
    if fault:
        with pytest.raises(ValueError):r.validate_bootstrap_launch38(path,path.parent)
    else:
        checked = r.validate_bootstrap_launch38(path,path.parent)
        assert checked['compiler_native_limit'] == checked['compiler_native'] == 0
        assert all(value == 0 for value in checked['model_limits'].values())


@pytest.mark.parametrize('fault',[None,'old_model','state','bad_tolerance','nan','probe','qualification'])
def test_load_proof_binds_actual_numpy_bundle_and_original_probe_without_forward(monkeypatch,fault):
    helper = SimpleNamespace(MODEL_SHA='a'*64,METADATA_SHA='b'*64,POLICY_STATE_SHA='c'*64)
    exports = {'actor.npz':{'sha256':'d'*64}}
    monkeypatch.setattr(d.r.backend,'helper36',lambda:helper)
    monkeypatch.setattr(d.r.backend,'export_identities38',lambda:exports)
    proof = {'schema':'b22-numpy38-load-proof-v1','execution_contract_id':r.CONTRACT,'backend':'NumPy',
        'model_sha256':'a'*64,'metadata_sha256':'b'*64,'source_policy_state_sha256':'c'*64,
        'export36_artifacts':exports,'probe_rows':32,'probe_prediction_calls':1,'probe_actions_within_tolerance':True,
        'probe_actions_byte_exact':False,'absolute_tolerance':1e-6,'verified_without_engine_or_reset':True,
        'control_qualification_granted':False,'ppo_load_calls':0,'torch_load_calls':0,
        'clipped16_max_absolute_error':1e-7,'half_gain_max_absolute_error':1e-7}
    if fault == 'old_model':proof['model_sha256'] = 'e'*64
    elif fault == 'state':proof['source_policy_state_sha256'] = 'e'*64
    elif fault == 'bad_tolerance':proof['half_gain_max_absolute_error'] = 2e-6
    elif fault == 'nan':proof['clipped16_max_absolute_error'] = float('nan')
    elif fault == 'probe':proof['probe_prediction_calls'] = True
    elif fault == 'qualification':proof['control_qualification_granted'] = True
    if fault:
        with pytest.raises(ValueError):d.verify_load_proof(proof,{'checkpoint_sha256':'a'*64},{'export36':exports})
    else:d.verify_load_proof(proof,{'checkpoint_sha256':'a'*64},{'export36':exports})


def cases():
    reports,references = [],{}
    for stage in r.stages():
        score = {'case_id':stage['case_id'],'seed':stage['seed'],'mirror':False,'actor':'final_policy',
            'windows':{'control_cap':1600,'hold':stage['hold'],'drive':[175,stage['release_tick']]},
            'record':{'full_horizon':True},'completed_controls':1600,'task_passed':True,
            'rl_terms':{'drive_complete':True,'sse_total':2.,'torque_cost_sum':8.}}
        before = {'stage':stage['name'],'pairing_identity':{'actual_initial_state':'same'},'cases':[score]}
        references[stage['name']] = before
        reports.append(copy.deepcopy(before))
    return reports,references


@pytest.mark.parametrize('fault',[None,'sse','cost','task','short','seed','initial','missing','reordered'])
def test_two_full_original_tasks_and_metric_upper_factor_are_the_predeclared_result_gate(fault):
    reports,references = cases()
    if fault == 'sse':reports[0]['cases'][0]['rl_terms']['sse_total'] = 2.0021
    elif fault == 'cost':reports[0]['cases'][0]['rl_terms']['torque_cost_sum'] = 8.0081
    elif fault == 'task':reports[0]['cases'][0]['task_passed'] = False
    elif fault == 'short':reports[0]['cases'][0].update(completed_controls=100,record={'full_horizon':False})
    elif fault == 'seed':reports[0]['cases'][0]['seed'] += 1
    elif fault == 'initial':reports[0]['pairing_identity'] = {'foreign':True}
    elif fault == 'missing':reports.pop()
    elif fault == 'reordered':reports.reverse()
    if fault in ('seed','initial','missing','reordered'):
        with pytest.raises(ValueError):d.compare_reports(reports,references)
    else:
        value = d.compare_reports(reports,references)
        assert value['backend_compatibility_passed'] is (fault is None)
        assert value['qualification_granted'] is False


def test_whole_native_ledger_is_retained_except_two_load_expectations(monkeypatch):
    import ast
    prior = r.old35.pinned('test_numpy38_pinned31',r.ROOT/'scripts/read_b22_suite31.py')
    # Namespace itself needs only require for the independently compiled function.
    base = SimpleNamespace(require=r.c.require)
    original_parse = ast.parse
    original_copy = copy.deepcopy
    text = (r.ROOT/'runtime/d1_b22/research/continuation26/root26/reader26.py').read_text()
    original_node = next(n for n in original_parse(text).body if isinstance(n,ast.FunctionDef) and n.name == 'check_ledgers')
    captured = []
    def capture(text):
        tree = original_parse(text)
        captured.append(tree)
        return tree
    monkeypatch.setattr(d.ast,'parse',capture)
    monkeypatch.setattr(d.copy,'deepcopy',lambda node:node)
    function = d.native_ledger_checker(base)
    assert function.__name__ == 'check_ledgers'
    changed_node = next(n for n in captured[-1].body if isinstance(n,ast.FunctionDef) and n.name == 'check_ledgers')
    before = next(n for n in ast.walk(original_node) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id == 'expected' for t in n.targets))
    after = next(n for n in ast.walk(changed_node) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id == 'expected' for t in n.targets))
    for old,new in zip(before.value.keywords,after.value.keywords):
        if old.arg in ('load','torch_load'):
            assert isinstance(new.value,ast.Constant) and type(new.value.value) is int and new.value.value == 0
            new.value = original_copy(old.value)
    assert ast.dump(original_node,include_attributes=False) == ast.dump(changed_node,include_attributes=False)
    assert d.metadata_checker(prior).__name__ == 'script_sources'


def test_owned_cleanup_records_signal_instead_of_starting_more_work():
    previous = signal.getsignal(signal.SIGTERM)
    host = SimpleNamespace(clean_descendants=lambda parent,process:(signal.raise_signal(signal.SIGTERM) or {'remaining':{}}))
    with pytest.raises(InterruptedError,match='during owned cleanup'):r.cleanup_owned(host,None)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_closed_fact_subtotal_cannot_double_count_active():
    stage = r.stages()[0]
    counts = {'controls':1600,'normal_native':8000,'compiler_native':2}
    assert r.counted({stage['name']:counts},stage,counts) == counts


@pytest.fixture
def execution(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    monkeypatch.setattr(r,'REGISTERED_ROOT',tmp_path)
    output = tmp_path/'results/new38'
    output.parent.mkdir()
    monkeypatch.setattr(r,'OUTPUT',output)
    monkeypatch.setattr(r.os,'getpriority',lambda *args:10)
    state = SimpleNamespace(memory=8 << 30,now=0.,calls=[],cleaned=[],fault=None)
    monkeypatch.setattr(r.supervision,'memory',lambda:state.memory)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda path:SimpleNamespace(free=8 << 30))
    monkeypatch.setattr(r.time,'sleep',lambda seconds:setattr(state,'now',state.now+seconds))
    hostfile = tmp_path/'host.py'
    hostfile.write_text('# fake only')
    readerfile = tmp_path/'scripts/read_b22_numpy38.py'
    readerfile.parent.mkdir()
    readerfile.write_text('# fake reader')
    (tmp_path/'scripts/run_b22_gain_suite35.py').write_text('# fake closure identity only')
    source = {'git_commit':'a'*40,'authored':{str(readerfile):r.c.identity(readerfile)}}
    config = {'host':str(hostfile),'source_hashes':{str(hostfile):r.c.identity(hostfile)},
        'runtime_environment':{'LD_BIND_NOW':'1','LD_LIBRARY_PATH':None},
        'shared_session':{'library':'/fake-library','binding_module':'/fake-binding','compiler_native_limit':2}}
    configpath = tmp_path/r.c.MANIFEST
    configpath.parent.mkdir(parents=True,exist_ok=True)
    r.c.write_json(configpath,config)
    refs = {'inputs':{},'old_inputs':{}}
    host = SimpleNamespace(subreaper=lambda:state.calls.append('subreaper'),
        clean_descendants=lambda parent,process:state.cleaned.append(process) or {'remaining':{}})
    monkeypatch.setattr(r.c,'load_pinned',lambda *args:host)
    monkeypatch.setattr(r,'load_file',lambda *args:SimpleNamespace(verify_bootstrap=lambda *args:True))
    monkeypatch.setattr(r,'case_completion',lambda stage,worker:worker['result']['completed_controls'])
    def supervisor(argv,log,**kwargs):
        if '--bootstrap' in argv:
            name = 'bootstrap'
            folder = Path(argv[argv.index('--output')+1])
            r.c.write_json(folder/'bootstrap38_report.json',{'status':'COMPLETE'})
        elif '--request' in argv:
            request = r.c.read_json(argv[-1])
            name = request['arm']
            folder = Path(request['spec']['output_directory'])
            folder.mkdir()
            session = {'execution_contract_id':r.CONTRACT,'output_directory':str(folder),
                'gain38':.5,'source_hashes':r.c.read_json(output/'release.json')['inputs']}
            r.c.write_json(folder/'session.json',session)
            worker = {'schema':'b22-numpy38-worker-receipt-v1','execution_contract_id':r.CONTRACT,
                'session_identity':r.c.identity(folder/'session.json'),
                'result':{'completed_controls':1600,'python':{'control_completed':1600,'native_returned':8000},
                          'C_final':{'construction_returns':2,'control_returns':8000}}}
            r.c.write_json(folder/'worker_receipt.json',worker)
        elif '--stage' in argv:
            name = argv[argv.index('--stage')+1]+'_reader'
            stage = name.removesuffix('_reader')
            saved = {'schema':'b22-numpy38-stage-readback-v1','status':'COMPLETE','execution_contract_id':r.CONTRACT,
                'stage':stage,'release_identity':r.c.identity(output/'release.json'),'reader_identity':r.c.identity(readerfile),
                'actual_counts':{'controls':1600,'normal_native':8000,'compiler_native':2},'gain_ledger_rows':1600,
                'numpy_predict_calls':1601,'numpy_predict_rows':1632,'ppo_load':0,'torch_load':0}
            r.c.write_json(Path(argv[-1]),saved)
        else:
            name = 'comparison'
            r.c.write_json(output/'comparison.json',{'schema':'b22-numpy38-comparison-v1','status':'COMPLETE',
                'release_identity':r.c.identity(output/'release.json'),
                'stage_reports':{stage['name']+'_readback.json':r.c.identity(output/(stage['name']+'_readback.json')) for stage in r.stages()},
                'new_actual_counts':{'controls':3200,'normal_native':16000,'compiler_native':4},
                'all_two_full_horizons':True,'all_two_tasks_passed':True,'backend_compatibility_passed':True,'qualification_granted':False})
        state.calls.append(name)
        if state.fault == name:return 1
        if state.fault == 'resource' and name == 'bootstrap':state.memory = (4 << 30)-1
        if state.fault == 'late' and name == 'bootstrap':state.now = 901.
        return 0
    state.kwargs = {'output':output,'source_reader':lambda:source,'references_reader':lambda:refs,
        'exports_reader':dict,'config_reader':lambda:(config,{}),'supervisor':supervisor,
        'validator':lambda path:None,'clock':lambda:state.now}
    state.run = lambda:r.execute(**state.kwargs)
    return state


def test_fake_lifecycle_cold_bootstrap_then_two_numpy_physical_cases_and_saved_readback(execution):
    value = execution.run()
    assert value['status'] == 'COMPLETE' and value['failure'] is None
    assert execution.calls == ['subreaper','bootstrap','flat_1p6_B','flat_1p6_B_reader','yaw_1p2_B','yaw_1p2_B_reader','comparison']
    assert value['actual_counts'] == {'controls':3200,'normal_native':16000,'compiler_native':4}
    assert value['actual_nn_counts'] == {'predict_calls':3202,'predict_rows':3264,'ppo_load':0,'torch_load':0}
    assert not value['qualification_granted'] and value['backend_compatibility_passed']
    assert execution.cleaned and all(item is None for item in execution.cleaned)
    with pytest.raises(ValueError,match='no retry'):execution.run()


@pytest.mark.parametrize('fault',['bootstrap','flat_1p6_B','flat_1p6_B_reader','yaw_1p2_B','comparison','resource','late'])
def test_engineering_failure_never_retries_and_unknown_actual_is_not_zero(execution,fault):
    execution.fault = fault
    value = execution.run()
    assert value['status'] == 'INCOMPLETE' and value['actual_counts'] is None and value['actual_nn_counts'] is None
    assert execution.cleaned and all(item is None for item in execution.cleaned)
    assert execution.calls.count('bootstrap') == 1
    if fault in ('bootstrap','resource','late'):
        assert 'flat_1p6_B' not in execution.calls and value['known_actual_counts']['controls'] == 0
    elif fault in ('flat_1p6_B','flat_1p6_B_reader'):
        assert 'yaw_1p2_B' not in execution.calls
        assert value['known_actual_counts'] == {'controls':1600,'normal_native':8000,'compiler_native':2}


def test_initial_admission_below6_does_not_claim_output_or_spawn(execution):
    execution.memory = (6 << 30)-1
    value = execution.run()
    assert value['status'] == 'NOT_READY' and not execution.calls and not r.OUTPUT.exists()


def test_no_live_child_waits_for6_before_creating_next_request(execution,monkeypatch):
    original = execution.kwargs['supervisor']
    def supervisor(argv,*args,**kwargs):
        code = original(argv,*args,**kwargs)
        if '--bootstrap' in argv:execution.memory = 4 << 30
        return code
    observed = []
    def sleep(seconds):
        journal = r.c.read_json(r.OUTPUT/'report.json')
        assert journal['active_stage'] is None and journal['nonce'] is None
        assert not (r.OUTPUT/'flat_1p6_B').exists() and not (r.OUTPUT/'flat_1p6_B_launch').exists()
        observed.append(seconds)
        execution.now += seconds
        execution.memory = int(5.84*(1 << 30)) if execution.now < 10 else 6 << 30
    execution.kwargs['supervisor'] = supervisor
    monkeypatch.setattr(r.time,'sleep',sleep)
    value = execution.run()
    assert value['status'] == 'COMPLETE' and observed == [5.,5.]
    wait = value['admission_waits'][0]
    assert wait['stage'] == 'flat_1p6_B' and wait['status'] == 'ADMITTED' and wait['duration_s'] == 10.
    assert execution.calls.count('flat_1p6_B') == 1


@pytest.mark.parametrize('fault',['timeout','below4','unknown','signal','remaining_window','late_source'])
def test_admission_wait_stops_without_claim_or_retry(execution,monkeypatch,fault):
    original = execution.kwargs['supervisor']
    def supervisor(argv,*args,**kwargs):
        code = original(argv,*args,**kwargs)
        if '--bootstrap' in argv:
            execution.memory = int(5.84*(1 << 30))
            if fault == 'remaining_window':execution.now = 600.
        return code
    def sleep(seconds):
        if fault == 'signal':signal.raise_signal(signal.SIGTERM)
        execution.now += seconds
        if fault == 'below4':execution.memory = (4 << 30)-1
        elif fault == 'unknown':execution.memory = None
        elif fault == 'late_source':execution.memory = 6 << 30
    if fault == 'late_source':
        def validator(path):
            if execution.now >= 5:execution.now = 600.
        execution.kwargs['validator'] = validator
    execution.kwargs['supervisor'] = supervisor
    monkeypatch.setattr(r.time,'sleep',sleep)
    value = execution.run()
    assert value['status'] == 'INCOMPLETE' and 'flat_1p6_B' not in execution.calls
    assert not (r.OUTPUT/'flat_1p6_B_launch').exists() and not (r.OUTPUT/'flat_1p6_B').exists()
    assert value['known_actual_counts'] == {'controls':0,'normal_native':0,'compiler_native':0}
    if fault != 'remaining_window':assert value['admission_waits'][0]['status'] == 'STOPPED'
    if fault == 'timeout':assert value['admission_waits'][0]['duration_s'] == 300.
