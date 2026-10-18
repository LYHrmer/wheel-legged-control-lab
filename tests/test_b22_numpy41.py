"""41 AST/source/metadata and fake owner tests; no NN, bootstrap or physics."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('test_numpy41',ROOT/'scripts/run_b22_numpy41.py')
d=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(d)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setattr(d,'reused40',lambda:{'inputs':{},'report_identity':{'bytes':1,'sha256':'e'*64},'report':{}})
    return d.build_scope()


def test_default_plan_is_static_zero_execution_and_yaw_only(monkeypatch,capsys):
    monkeypatch.setattr(d,'scope',lambda:pytest.fail('entered source/runtime graph'))
    assert d.main([])==0
    value=json.loads(capsys.readouterr().out)
    assert value['status']=='PLAN_ONLY' and value['stage_order']==['bootstrap','yaw_1p2_B','saved_readback','comparison']
    assert value['new_actual_budget']=={'controls':1600,'normal_native':8000,'compiler_native':2}
    assert value['new_nn_budget']=={'predict_calls':1601,'predict_rows':1632,'ppo_load':0,'torch_load':0}
    assert not value['execution_started'] and not value['qualification_granted'] and not value['retry_permitted']


def test_factory_inverse_lock_shared_backend_and_precise_new_identity(graph):
    r=graph.runner
    assert graph.factory41_proof['inverse_ast_exact'] and graph.factory41_proof['all_sha_constants_unchanged']
    assert r.CONTRACT==graph.backend.CONTRACT==graph.worker.CONTRACT==d.CONTRACT
    assert r.OUTPUT==d.OUTPUT and r.AUTHORED==d.AUTHORED
    assert r.backend is graph.worker.backend is graph.backend
    assert sys.modules['b22_numpy_backend41'] is graph.backend
    assert graph.reader.r is r
    assert [s['name'] for s in r.stages()]==['yaw_1p2_B']
    assert r.LIMITS['controls']==1600 and r.LIMITS['numpy_predict_calls']==1601
    assert r.LIMITS['worker_sessions']==2 and r.LIMITS['reserve_effective_bytes']==4<<30
    assert r.execute.__kwdefaults__['references_reader'] is r.reference35
    assert r.execute.__kwdefaults__['validator'] is r.validate_release
    assert {k:r.LIMITS[k] for k in ('bootstrap_wall_s','stage_wall_s','reader_wall_s','total_wall_s')}=={
        'bootstrap_wall_s':120,'stage_wall_s':360,'reader_wall_s':240,'total_wall_s':900}
    assert graph.request_wall41_proof['inverse_ast_exact']
    assert all(m.__file__==str(d.DRIVER) for m in (graph.worker,r,graph.reader,graph.backend))
    source=graph.worker.derived_source38()
    assert 'from b22_numpy_backend41 import NumpyCounter38 as ModelCounter' in source
    factory,_=d.factory41();tree,_=factory.derive('reader')
    values=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert any("expected_origins['b22_numpy_backend41'] = str(c.ROOT/'scripts/run_b22_numpy41.py')" in x for x in values)
    assert any(x=="'scripts/run_b22_numpy41.py'" for x in values)
    assert not any('scripts/run_b22_numpy39.py' in x for x in values)


def test_new_reader_retains40_exact_empty_maps_not_the_broken_nonempty_torch_premise(graph):
    factory,_=d.factory41()
    maps=d.load40().install_adapter40(factory,graph)
    assert maps['inverse_ast_exact'] and 'exact empty paths' in maps['only_semantic_seam']
    text=(ROOT/'scripts/read_b22_suite31.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='script_sources')
    changed=d.load40().adapt_maps40(ast.get_source_segment(text,node))
    assert "document40 == {'paths': []}" in changed
    assert 'mapped and all' not in changed


@pytest.mark.parametrize('argv,owner',[
    (['--session','/s','--output','/o'],'worker'),
    (['--bootstrap','--session','/s','--output','/o'],'worker'),
    (['--release','/r','--stage','yaw_1p2_B','--report','/o'],'reader'),
    (['--release','/r','--compare','--report','/o'],'reader')])
def test_exact_host_worker_and_saved_reader_argv(monkeypatch,argv,owner):
    seen=[];graph=SimpleNamespace(worker=SimpleNamespace(main=lambda a:seen.append(('worker',a)) or 0),
        reader=SimpleNamespace(main=lambda a:seen.append(('reader',a)) or 0))
    monkeypatch.setattr(d,'scope',lambda:graph)
    assert d.main(argv)==0 and seen==[(owner,argv)]


def case(stage,controls=1600):
    return {'case_id':stage['case_id'],'seed':stage['seed'],'mirror':False,'actor':'final_policy',
        'windows':{'control_cap':1600,'hold':stage['hold'],'drive':[175,stage['release_tick']]},
        'record':{'full_horizon':controls==1600},'completed_controls':controls,'task_passed':controls==1600,
        'rl_terms':{'drive_complete':controls==1600,'sse_total':2.,'torque_cost_sum':8.}}


@pytest.mark.parametrize('controls',[1600,599])
def test_combined_analysis_preserves_scopes_and_counts_short_yaw_honestly(graph,monkeypatch,tmp_path,controls):
    r=graph.runner;factory,_=d.factory41();all_stages=factory.build_scope().runner.stages()
    refs={s['name']:{'stage':s['name'],'pairing_identity':{'same':s['name']},'cases':[case(s)]} for s in all_stages}
    old=copy.deepcopy(refs['flat_1p6_B']);new=copy.deepcopy(refs['yaw_1p2_B']);new['cases']=[case(all_stages[1],controls)]
    new['actual_counts']={'controls':controls,'normal_native':5*controls,'compiler_native':2};new['artifacts']={}
    new.update(numpy_predict_calls=controls+1,numpy_predict_rows=controls+32)
    reused={'report':old,'report_identity':{'bytes':1,'sha256':'e'*64}}
    reuse_calls=[]
    monkeypatch.setattr(d,'reused40',lambda:reuse_calls.append(True) or reused)
    monkeypatch.setattr(r,'reference35',lambda:{'reports':refs})
    def validate(path):
        # A real validation reads this through plan and closure. The surrounding
        # compare must use the same call-local snapshot, not hash it three times.
        graph.current_reuse41();graph.current_reuse41()
        return {'plan':{'output':str(tmp_path)}}
    monkeypatch.setattr(r,'validate_release',validate)
    monkeypatch.setattr(r,'stage_counts',lambda *a:new['actual_counts'])
    path=tmp_path/'release.json';write(path,{})
    write(tmp_path/'yaw_1p2_B_readback.json',new)
    out=graph.reader.compare(path)
    assert out['new_actual_counts']==new['actual_counts']
    assert out['reused_validated_case_counts']==d.PHYSICAL
    assert out['combined_validated_case_counts']['controls']==1600+controls
    assert out['original39_actual_counts'] is None and not out['historical_total_resources_known']
    assert out['reused_execution_contract_id']=='C39_B22_numpy_half_gain_v1'
    assert out['backend_compatibility_passed'] is (controls==1600)
    assert not out['qualification_granted']
    assert len(reuse_calls)==1
    graph.reader.compare(path)
    assert len(reuse_calls)==2  # A separate comparison revalidates old evidence.


def test_reuse_snapshot_is_call_local_copied_and_cleared_on_error(graph,monkeypatch):
    calls=[];state={'fail':False}
    def reuse():
        calls.append(True)
        if state['fail']:raise ValueError('changed old evidence')
        return {'inputs':{},'generation':len(calls)}
    monkeypatch.setattr(d,'reused40',reuse)
    with graph.reuse_snapshot41():
        first=graph.current_reuse41();first['generation']=-1
        with graph.reuse_snapshot41():assert graph.current_reuse41()['generation']==1
    with graph.reuse_snapshot41():assert graph.current_reuse41()['generation']==2
    state['fail']=True
    with pytest.raises(ValueError,match='changed old evidence'),graph.reuse_snapshot41():graph.current_reuse41()
    state['fail']=False
    with graph.reuse_snapshot41():assert graph.current_reuse41()['generation']==4
    assert graph.current_reuse41()['generation']==5  # No cache outside a call.


@pytest.fixture
def saved40(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'ROOT',tmp_path)
    output=tmp_path/'old40';original=tmp_path/'old39';original.mkdir();output.mkdir()
    flat=original/'flat_1p6_B';flat.mkdir();raw=flat/'raw.json';write(raw,{'actual':'old39'})
    source40={}
    for name in d.SOURCE40:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fake_source')
        source40[str(p)]=d.identity(p)
    source={'git_commit':d.SOURCE_COMMIT40,'authored':source40};proof={'inverse_ast_exact':True}
    old_release={'source':{'old39':True},'inputs':{}}
    def tree(root):return {str(p):d.identity(p) for p in root.rglob('*') if p.is_file()}
    inputs={'schema':'b22-numpy40-inputs-v1','source40':source,'source39':old_release['source'],
        'original39_bound_inputs':{},'frozen39_payloads':tree(original),'derivation40':copy.deepcopy(proof)}
    write(output/'inputs.json',inputs)
    report={'schema':'b22-numpy40-saved-readback-v1','status':'COMPLETE','failure':None,'execution_contract_id':'C40',
        'analyzed_execution_contract_id':'C39_B22_numpy_half_gain_v1','source40':source,'inputs_identity':d.identity(output/'inputs.json'),
        'stage':'flat_1p6_B','original39_parent_status':'INCOMPLETE','original39_actual_counts':None,'original39_actual_nn_counts':None,
        'yaw_evaluated':False,'qualification_granted':False,'new_actual_counts':dict.fromkeys(('controls','normal_native','compiler_native','predict_calls','predict_rows','ppo_load','torch_load'),0),
        'verified_saved_execution_counts':dict(d.PHYSICAL),'verified_saved_nn_counts':dict(d.NN),'derivation40':copy.deepcopy(proof),
        'cases':[{'seed':181202,'completed_controls':1600,'task_passed':True}],
        'artifacts':{'raw.json':d.identity(raw)}}
    write(output/'report.json',report)
    reader=SimpleNamespace(CONTRACT='C40',ZERO=report['new_actual_counts'],ORIGINAL=original,
        read_json=lambda p:json.loads(p.read_text()),check_original39=lambda:(old_release,{}),tree_identity=tree,
        load_graph39=lambda:(None,None),install_adapter40=lambda *a:copy.deepcopy(proof))
    monkeypatch.setattr(d,'REUSED',output);monkeypatch.setattr(d,'load40',lambda:reader)
    monkeypatch.setattr(d,'pinned_sources',dict)
    anchors={name:d.identity(output/name)['sha256'] for name in ('report.json','inputs.json')}
    monkeypatch.setattr(d,'REUSED_SHA',anchors)
    return output,original,report,inputs,anchors


def rewrite40(saved40):
    output,_,report,inputs,anchors=saved40
    write(output/'inputs.json',inputs);report['inputs_identity']=d.identity(output/'inputs.json')
    write(output/'report.json',report)
    anchors.update({name:d.identity(output/name)['sha256'] for name in anchors})


def test_reuse_is_saved40_identity_not_new41_physical_spending(saved40):
    value=d.reused40()
    assert value['reused_validated_case_counts']==d.PHYSICAL
    assert value['original39_actual_counts'] is None and value['original39_actual_nn_counts'] is None
    assert value['execution_contract_id']=='C39_B22_numpy_half_gain_v1' and value['adjudication_contract_id']=='C40'


@pytest.mark.parametrize('fault',['status','newphysical','typedzero','nn','oldtotal','source','inputid','raw','escape','case','proof'])
def test_reuse_rejects_incomplete_foreign_typed_or_changed_old_artifacts(saved40,fault):
    output,original,report,_inputs,_=saved40
    if fault=='status':report['status']='INCOMPLETE'
    elif fault=='newphysical':report['new_actual_counts']['controls']=1
    elif fault=='typedzero':report['new_actual_counts']['torch_load']=False
    elif fault=='nn':report['verified_saved_nn_counts']['predict_rows']=1631
    elif fault=='oldtotal':report['original39_actual_counts']=dict(d.PHYSICAL)
    elif fault=='source':report['source40']['git_commit']='f'*40
    elif fault=='inputid':report['inputs_identity']={'bytes':0,'sha256':'f'*64}
    elif fault=='raw':write(original/'flat_1p6_B/raw.json',{'tampered':True})
    elif fault=='escape':report['artifacts']['../escape']=report['artifacts']['raw.json']
    elif fault=='case':report['cases'][0]['completed_controls']=1599
    else:report['derivation40']['inverse_ast_exact']=False
    rewrite40(saved40)
    if fault=='inputid':report['inputs_identity']={'bytes':0,'sha256':'f'*64};write(output/'report.json',report)
    with pytest.raises(ValueError):d.reused40()


def test_reuse_does_not_cherry_pick_only_task_positive_cases(saved40):
    saved40[2]['cases'][0]['task_passed']=False;rewrite40(saved40)
    assert d.reused40()['report']['cases'][0]['task_passed'] is False


@pytest.fixture
def execution(graph,monkeypatch,tmp_path):
    r=graph.runner
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'REGISTERED_ROOT',tmp_path)
    output=tmp_path/'new41';monkeypatch.setattr(r,'OUTPUT',output)
    script=tmp_path/'scripts/run_b22_numpy41.py';script.parent.mkdir();script.write_text('# fake')
    (tmp_path/'scripts/run_b22_gain_suite35.py').write_text('# fake')
    config={'host':str(tmp_path/'host.py'),'source_hashes':{},'runtime_environment':{},
        'shared_session':{'library':'/lib','binding_module':'/binding','compiler_native_limit':2}}
    Path(config['host']).write_text('# fake');config['source_hashes'][config['host']]=r.c.identity(config['host'])
    path=tmp_path/r.c.MANIFEST;path.parent.mkdir(parents=True,exist_ok=True);write(path,config)
    source={'git_commit':'a'*40,'authored':{str(script):r.c.identity(script)}}
    state=SimpleNamespace(memory=8<<30,now=0.,calls=[],fault=None)
    monkeypatch.setattr(r.os,'getpriority',lambda *a:10)
    monkeypatch.setattr(r.supervision,'memory',lambda:state.memory)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda p:SimpleNamespace(free=8<<30))
    monkeypatch.setattr(r.time,'sleep',lambda seconds:setattr(state,'now',state.now+seconds))
    monkeypatch.setattr(graph.package,'inventory39',lambda:{'inputs':{}})
    monkeypatch.setattr(r,'_inventory39',graph.package.inventory39)
    monkeypatch.setattr(r,'_scope_reader',lambda:SimpleNamespace(verify_bootstrap=lambda *a:True))
    host=SimpleNamespace(subreaper=lambda:None,clean_descendants=lambda *a:{'remaining':{}})
    monkeypatch.setattr(r.c,'load_pinned',lambda *a:host)
    monkeypatch.setattr(r,'case_completion',lambda *a:None)
    def supervisor(argv,log,**kwargs):
        if '--bootstrap' in argv:
            name='bootstrap';write(output/'bootstrap/bootstrap38_report.json',{'status':'COMPLETE'})
        elif '--request' in argv:
            name='yaw_1p2_B';request=r.c.read_json(Path(argv[-1]));run=Path(request['spec']['output_directory']);run.mkdir()
            session={'execution_contract_id':d.CONTRACT,'output_directory':str(run),'gain38':.5,
                     'source_hashes':r.c.read_json(output/'release.json')['inputs']}
            write(run/'session.json',session)
            write(run/'worker_receipt.json',{'schema':'b22-numpy41-worker-receipt-v1','execution_contract_id':d.CONTRACT,
                'session_identity':r.c.identity(run/'session.json'),'result':{'python':{'control_completed':1600,'native_returned':8000},
                    'C_final':{'construction_returns':2,'control_returns':8000}}})
        elif '--stage' in argv:
            name='reader';write(Path(argv[-1]),{'schema':'b22-numpy41-stage-readback-v1','status':'COMPLETE','execution_contract_id':d.CONTRACT,
                'stage':'yaw_1p2_B','release_identity':r.c.identity(output/'release.json'),'reader_identity':r.c.identity(script),
                'actual_counts':dict(d.PHYSICAL),'gain_ledger_rows':1600,'numpy_predict_calls':1601,'numpy_predict_rows':1632,'ppo_load':0,'torch_load':0})
        else:
            name='comparison';write(output/'comparison.json',{'schema':'b22-numpy41-comparison-v1','status':'COMPLETE',
                'release_identity':r.c.identity(output/'release.json'),'stage_reports':{'yaw_1p2_B_readback.json':r.c.identity(output/'yaw_1p2_B_readback.json')},
                'new_actual_counts':dict(d.PHYSICAL),'qualification_granted':False,'all_two_full_horizons':True,
                'all_two_tasks_passed':True,'backend_compatibility_passed':True})
        state.calls.append(name)
        if state.fault==name:return 1
        if state.fault=='ram' and name=='bootstrap':state.memory=(4<<30)-1
        return 0
    state.run=lambda:r.execute(output=output,source_reader=lambda:source,references_reader=lambda:{'inputs':{},'old_inputs':{}},
        exports_reader=dict,config_reader=lambda:(config,{}),supervisor=supervisor,validator=lambda p:None,clock=lambda:state.now)
    return state,r,output


def test_only_bootstrap_and_missing_yaw_spawn_once_no_flat_retry(execution):
    state,_r,output=execution;value=state.run()
    assert value['status']=='COMPLETE' and value['actual_counts']==d.PHYSICAL and value['actual_nn_counts']==d.NN
    assert state.calls==['bootstrap','yaw_1p2_B','reader','comparison']
    request=json.loads((output/'yaw_1p2_B_launch/request.json').read_text())
    go=json.loads((output/'yaw_1p2_B_launch/source_go.json').read_text())
    assert request['spec']==go['arms']['yaw_1p2_B']
    assert {k:request['spec'][k] for k in ('soft_s','close_s','hard_s')}=={'soft_s':300,'close_s':355,'hard_s':360}
    assert request['spec']['control_limit']==1600 and request['spec']['model_limits']['predict']==1601
    assert not (output/'flat_1p6_B').exists()
    with pytest.raises(ValueError,match='no retry'):state.run()


def test_request_ast_changes_only_three_wall_values(graph):
    factory,_=d.factory41();tree,_=factory.derive('runner')
    original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='script_request')
    changed=copy.deepcopy(original);seen=[]
    for node in ast.walk(changed):
        if isinstance(node,ast.Dict):
            for key,value in zip(node.keys,node.values):
                if isinstance(key,ast.Constant) and key.value in ('soft_s','close_s','hard_s'):
                    value.value={'soft_s':300,'close_s':355,'hard_s':360}[key.value];seen.append(key.value)
    import hashlib
    assert sorted(seen)==['close_s','hard_s','soft_s']
    assert graph.request_wall41_proof['derived_ast_sha256']==hashlib.sha256(ast.dump(changed,include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize('fault',['bootstrap','yaw_1p2_B','reader','comparison','ram'])
def test_any_engineering_failure_stops_without_retry_and_preserves_unknown(execution,fault):
    state,_r,_output=execution;state.fault=fault;value=state.run()
    assert value['status']=='INCOMPLETE' and value['actual_counts'] is None and value['actual_nn_counts'] is None
    assert value['cleanup']['remaining']=={} and state.calls.count('yaw_1p2_B')<=1
    if fault in ('bootstrap','ram'):assert state.calls==['bootstrap']
    if fault=='reader':assert value['known_actual_counts']==d.PHYSICAL


def test_initial_ram_not_ready_does_not_claim_independent_output(execution):
    state,_r,output=execution;state.memory=(6<<30)-1
    value=state.run()
    assert value['status']=='NOT_READY' and not state.calls and not output.exists()


def test_cold_parent_bootstrap_args_point_to_actual41_owner(graph):
    args=graph.runner.bootstrap_argv(d.OUTPUT/'bootstrap/session.json',d.OUTPUT/'bootstrap')
    assert args[2]==str(d.DRIVER) and '--bootstrap' in args
    before=signal.getsignal(signal.SIGTERM)
    def cleanup(*args):signal.raise_signal(signal.SIGTERM);return {'remaining':{}}
    with pytest.raises(InterruptedError):graph.runner.cleanup_owned(SimpleNamespace(clean_descendants=cleanup),None)
    assert signal.getsignal(signal.SIGTERM)==before



def test_fixed40_report_sha_rejects_an_arbitrary_new_complete_document(saved40):
    output,_,report,_,_=saved40
    report['unapproved_complete']=True;write(output/'report.json',report)
    with pytest.raises(ValueError,match='fixed C40 evidence changed'):d.reused40()


@pytest.mark.parametrize('fault',[None,'timeout','below4','unknown','late_window'])
def test_yaw_admission_wait_never_creates_or_retries_a_stage_early(execution,monkeypatch,fault):
    state,r,output=execution
    # Preserve the fake supervisor while changing only its post-bootstrap RAM.
    old_run=state.run
    # execute obtains all fixture dependencies through the closure of state.run;
    # temporarily wrap the existing supervisor keyword at the public boundary.
    original=r.execute
    def execute(**kwargs):
        supervisor=kwargs['supervisor']
        def wrapped(argv,*args,**options):
            code=supervisor(argv,*args,**options)
            if '--bootstrap' in argv:
                state.memory=int(5.84*(1<<30))
                if fault=='late_window':state.now=600.
            return code
        return original(**dict(kwargs,supervisor=wrapped))
    monkeypatch.setattr(r,'execute',execute)
    def sleep(seconds):
        assert not (output/'yaw_1p2_B_launch').exists() and not (output/'yaw_1p2_B').exists()
        journal=r.c.read_json(output/'report.json');assert journal['active_stage'] is None and journal['nonce'] is None
        state.now+=seconds
        if fault is None:state.memory=6<<30
        elif fault=='below4':state.memory=(4<<30)-1
        elif fault=='unknown':state.memory=None
    monkeypatch.setattr(r.time,'sleep',sleep)
    value=old_run()
    if fault is None:
        assert value['status']=='COMPLETE' and state.calls.count('yaw_1p2_B')==1
        assert value['admission_waits'][0]['duration_s']==5.
    else:
        assert value['status']=='INCOMPLETE' and state.calls==['bootstrap']
        assert not (output/'yaw_1p2_B_launch').exists()
