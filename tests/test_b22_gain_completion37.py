"""C37 cold metadata and fake ownership tests; never loads a policy or physics."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = load('test_completion37', 'scripts/run_b22_gain_completion37.py')
worker = load('test_owner37', 'scripts/run_b22_worker37.py')
reader = load('test_reader37', 'scripts/read_b22_gain_completion37.py')


def test_fixed_missing_matrix_budget_and_default_plan_are_cold(monkeypatch, capsys):
    monkeypatch.setattr(r, 'source_identity', lambda: pytest.fail('plan froze sources'))
    monkeypatch.setattr(r, 'load_reused35', lambda: pytest.fail('plan opened old evidence'))
    assert r.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert [s['name'] for s in value['stages']] == ['ramp_0p45_complete_zero','ramp_0p45_complete_B','yaw_1p2_mirror_zero','yaw_1p2_mirror_B']
    assert sum(s['controls'] for s in value['stages']) == 6800
    assert r.LIMITS['normal_native'] == 34000 and r.LIMITS['compiler_native'] == 8
    assert r.LIMITS['total_wall_s'] == 900 and r.LIMITS['reserve_effective_bytes'] == 4 << 30
    assert value['reused_old35_outer_actual_counts'] is None and value['historical_total_resources_known'] is False
    assert value['execution_started'] is False and value['training_calls'] == 0


def test_worker_entire_ast_changes_only_the_versioned_admission_literal():
    source, proof = worker.derived_source()
    old = r.load_old('test_old_worker35', 'scripts/run_b22_worker35.py').derived_sources()['script']
    restored = source.replace(repr(worker.CONTRACT), repr(r.old35.CONTRACT))
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(old))
    assert proof['execution_contract_id'] == worker.CONTRACT and source.count(repr(worker.CONTRACT)) == 1
    assert "np.float32(session[\'gain35\'])" in source


@pytest.mark.parametrize('stage', r.stages(), ids=lambda s:s['name'])
def test_worker_accepts_only_real_c37_missing_case_sessions(stage, tmp_path):
    spec = {**stage, 'arm':stage['name'],'execution_contract_id':worker.CONTRACT,'mode':'script','render':False,
            'qualification':False,'gain35':.5,'control_limit':stage['controls'],'output_directory':str(tmp_path)}
    worker.validate_session(spec, tmp_path)
    for key, value in [('execution_contract_id',r.old35.CONTRACT),('gain35',1.),('profile','flat_1p6'),('seed',0),('mirror',1)]:
        changed = dict(spec); changed[key] = value
        with pytest.raises(ValueError):worker.validate_session(changed,tmp_path)


def saved(stage, release_id=None, reader_id=None, schema='b22-gain-completion37-stage-readback-v1'):
    return {'schema':schema,'status':'COMPLETE','stage':stage['name'],'execution_contract_id':r.CONTRACT,
            'gain35':.5,'release_identity':release_id,'reader_identity':reader_id,'gain_ledger_rows':stage['controls'],
            'actual_counts':{'controls':stage['controls'],'normal_native':5*stage['controls'],'compiler_native':2}}


@pytest.mark.parametrize('fault', ['contract','stage','bool_count','native','gain','reader','compiler'])
def test_new_readbacks_cannot_masquerade_as_old_or_wrong_stage(fault):
    stage=r.stages()[0];value=saved(stage,{'release':'new'},{'reader':'new'})
    if fault=='contract':value['execution_contract_id']=r.old35.CONTRACT
    elif fault=='stage':value['stage']='flat_0p6_zero'
    elif fault=='bool_count':value['actual_counts']['compiler_native']=True
    elif fault=='native':value['actual_counts']['normal_native']+=1
    elif fault=='gain':value['gain35']=1.
    elif fault=='reader':value['reader_identity']={'other':True}
    else:value['actual_counts']['compiler_native']=3
    with pytest.raises(ValueError):r.counts(stage,value,{'release':'new'},{'reader':'new'})


def test_unique_new_counts_never_add_old_case_counts_or_double_finalize():
    stage=r.stages()[0];one={'controls':1800,'normal_native':9000,'compiler_native':2}
    assert r.counted({stage['name']:one},stage,one)==one
    assert r.counted({},stage,{'controls':1,'normal_native':3,'compiler_native':1})=={'controls':1,'normal_native':3,'compiler_native':1}
    assert r.REUSED_COUNTS=={'controls':18400,'normal_native':92000,'compiler_native':22}


@pytest.fixture
def archived35(tmp_path, monkeypatch):
    root=tmp_path;old=root/'old35';old.mkdir()
    monkeypatch.setattr(r,'ROOT',root);monkeypatch.setattr(r,'OLD_ROOT',old)
    source={'old':'source'}; reader_id={'bytes':3,'sha256':'a'*64}
    release={'source':source,'inputs':{str(root/'scripts/read_b22_gain_suite35.py'):reader_id}}
    monkeypatch.setattr(r.old35,'validate_release',lambda p:release)
    checked=[]
    monkeypatch.setattr(r.old35,'case_completion',lambda stage,receipt:checked.append(stage['name']))
    fake_worker=SimpleNamespace(derived_sources=dict,derivation_proof=lambda sources:{'C35':'unchanged'})
    monkeypatch.setattr(r,'load_old',lambda *args:fake_worker)
    reports={}
    for stage in r.old35.stages()[:-4]:
        run=old/stage['name'];run.mkdir()
        raw=run/'raw.bin';raw.write_bytes(b'preserved raw')
        r.c.write_json(run/'session.json',{'execution_contract_id':r.old35.CONTRACT,'gain35':.5,'output_directory':str(run),
            'argv':[str(root/'scripts/run_b22_worker35.py'),'--session',str(run/'session.json'),'--output',str(run)],
            'source_hashes':release['inputs']})
        r.c.write_json(run/'worker_receipt.json',{'session_identity':r.c.identity(run/'session.json')})
        r.c.write_json(run/'derived_sources35.json',{'C35':'unchanged'})
        item=saved(stage,None,reader_id,'b22-gain-suite35-stage-readback-v1')
        item.update(execution_contract_id=r.old35.CONTRACT,cases=[{'task_passed':False}],artifacts={'raw.bin':r.c.identity(raw)})
        reports[stage['name']]=item
    r.c.write_json(old/'release.json',release)
    for name,item in reports.items():
        item['release_identity']=r.c.identity(old/'release.json')
        r.c.write_json(old/(name+'_readback.json'),item)
    stage_map={name+'_readback.json':r.c.identity(old/(name+'_readback.json')) for name in reports}
    parent={'status':'INCOMPLETE','actual_counts':None,'active_stage_actual_counts':None,
        'known_actual_counts':r.REUSED_COUNTS,'cleanup':{'remaining':{}},'source':source,'stage_reports':stage_map}
    r.c.write_json(old/'report.json',parent)
    pins={name:r.c.identity(old/name) for name in ('release.json','report.json',*stage_map)}
    monkeypatch.setattr(r,'OLD_RECEIPTS',pins)
    return old,checked,parent


def test_reused_set_includes_every_failed_task_but_never_promotes_old_unknown_total(archived35):
    old,checked,parent=archived35
    value=r.load_reused35()
    assert len(value['reports'])==len(checked)==11
    assert all(x['cases'][0]['task_passed'] is False for x in value['reports'].values())
    assert value['validated_case_counts']==r.REUSED_COUNTS and value['old_outer_actual_counts'] is None
    assert r.c.read_json(old/'report.json')==parent


@pytest.mark.parametrize('fault',['parent_subset','parent_actual_zero','readback','raw','worker','added_positive_filter'])
def test_reuse_requires_exact_pinned_parent_map_and_all_original_payloads(archived35,fault):
    old,_,_=archived35
    if fault in ('parent_subset','parent_actual_zero','added_positive_filter'):
        path=old/'report.json';value=r.c.read_json(path)
        if fault=='parent_actual_zero':value['actual_counts']={'controls':0}
        else:value['stage_reports'].pop(next(iter(value['stage_reports'])))
        path.write_text(json.dumps(value))
    elif fault=='readback':(old/'flat_0p6_zero_readback.json').write_text('{}')
    elif fault=='worker':(old/'flat_0p6_zero/worker_receipt.json').write_text('{}')
    else:(old/'flat_0p6_zero/raw.bin').write_bytes(b'changed')
    with pytest.raises((ValueError,KeyError)):r.load_reused35()


def test_reader_derivatives_keep_numerics_and_old_step_reference_explicit():
    prior=r.old35.pinned('test_completion37_reader31',ROOT/'scripts/read_b22_suite31.py')
    source_check,canonical=reader.functions(prior,ROOT/'results/b22_gain_suite35_20261018_01/step_pair')
    assert 'C37_B22_half_gain_missing_cases_v1' in source_check.__code__.co_consts
    assert str(ROOT/'results/b22_gain_suite35_20261018_01/step_pair') in canonical.__code__.co_consts
    assert 'C35_B22_half_gain_fixed_suite_v1' not in source_check.__code__.co_consts


@pytest.fixture
def fake_execution(tmp_path, monkeypatch):
    root=tmp_path;output=root/'results/b22_gain_completion37_20261018_01';output.parent.mkdir()
    monkeypatch.setattr(r,'ROOT',root);monkeypatch.setattr(r,'REGISTERED_ROOT',root);monkeypatch.setattr(r,'OUTPUT',output)
    source={'git_commit':'a'*40,'authored':{}}
    monkeypatch.setattr(r,'source_identity',lambda:source)
    monkeypatch.setattr(r,'load_reused35',lambda:{'inputs':{},'old_release':{'inputs':{}},'status':'VALIDATED_CLOSED_CASES_ONLY',
        'old_outer_actual_counts':None,'historical_total_resources_known':False,'validated_case_counts':r.REUSED_COUNTS})
    (root/'scripts').mkdir();(root/'scripts/read_b22_gain_completion37.py').write_text('fake reader')
    (root/'host.py').write_text('fake host')
    rid=r.c.identity(root/'scripts/read_b22_gain_completion37.py')
    inputs={str(root/'scripts/read_b22_gain_completion37.py'):rid,str(root/'host.py'):r.c.identity(root/'host.py')}
    monkeypatch.setattr(r,'closure',lambda *args:inputs)
    monkeypatch.setattr(r,'validate_release',lambda path:r.c.read_json(path))
    monkeypatch.setattr(r.c,'registered_config',lambda:({'host':str(root/'host.py')},{}))
    monkeypatch.setattr(r.os,'getpriority',lambda *args:10)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda path:SimpleNamespace(free=4<<30))
    monkeypatch.setattr(r,'known_partial',lambda *args:None)
    monkeypatch.setattr(r,'case_completion',lambda *args:None)
    called=[];cleaned=[];now=[0.];fault=[None]
    monkeypatch.setattr(r.time,'monotonic',lambda:now[0])
    memory_calls=[0]
    def memory():
        memory_calls[0]+=1
        return (5<<30) if fault[0]=='admission' else (7<<29) if fault[0]=='reserve' and memory_calls[0]>1 else (8<<30)
    monkeypatch.setattr(r.supervision,'memory',memory)
    class Host:
        def subreaper(self):called.append('subreaper')
        def clean_descendants(self,*args):
            cleaned.append('own')
            return {'remaining':{1:2} if fault[0]=='cleanup' else {}}
    monkeypatch.setattr(r.c,'load_pinned',lambda *args:Host())
    def request(stage,*args):
        called.append(stage['name']);(output/stage['name']).mkdir()
        r.c.write_json(output/stage['name']/'worker_receipt.json',{})
        return ['worker',stage['name']]
    monkeypatch.setattr(r,'request',request)
    def supervise(argv,log,**kwargs):
        if argv[0]=='worker':
            if fault[0]=='worker_error':raise InterruptedError('synthetic worker interruption')
            if fault[0]=='wall':now[0]=901.;kwargs['sample'](False)
            return 0
        if '--stage' in argv:
            name=argv[argv.index('--stage')+1];stage=next(s for s in r.stages() if s['name']==name)
            r.c.write_json(Path(argv[-1]),saved(stage,r.c.identity(output/'release.json'),rid))
        else:
            new={s['name']+'_readback.json':r.c.identity(output/(s['name']+'_readback.json')) for s in r.stages()}
            total={'controls':6800,'normal_native':34000,'compiler_native':8}
            combined={k:total[k]+r.REUSED_COUNTS[k] for k in total}
            if fault[0]=='combined':combined['controls']=0
            r.c.write_json(Path(argv[-1]),{'schema':'b22-gain-completion37-comparison-v1','status':'COMPLETE',
                'release_identity':r.c.identity(output/'release.json'),'new_stage_reports':new,
                'new_actual_counts':total,'reused_validated_case_counts':r.REUSED_COUNTS,
                'combined_validated_case_counts':combined,'old35_outer_actual_counts':None,
                'historical_total_resources_known':False,'finite_suite_task_qualification_passed':False,
                'RL_contribution_passed':False,'no_task_regression':False})
        return 0
    monkeypatch.setattr(r.supervision,'supervise',supervise)
    return output,called,cleaned,fault


def test_parent_only_executes_four_new_sessions_and_never_claims_historical_total(fake_execution):
    output,called,cleaned,_=fake_execution
    value=r.execute(output)
    assert value['status']=='COMPLETE'
    assert called==['subreaper',*[s['name'] for s in r.stages()]] and cleaned==['own']
    assert value['actual_counts']=={'controls':6800,'normal_native':34000,'compiler_native':8}
    assert value['combined_validated_case_counts']=={'controls':25200,'normal_native':126000,'compiler_native':30}
    assert value['reused_old35_outer_actual_counts'] is None and value['historical_total_resources_known'] is False
    with pytest.raises(ValueError,match='no retry'):r.execute(output)
    assert (output/'report.json').is_file()


@pytest.mark.parametrize('fault',['admission','reserve','worker_error','wall','cleanup','combined'])
def test_owned_failure_or_unknown_counts_never_becomes_a_complete_new_attempt(fake_execution,fault):
    output,called,cleaned,setting=fake_execution;setting[0]=fault
    value=r.execute(output)
    if fault=='admission':
        assert value['status']=='NOT_READY' and not output.exists() and not called
    else:
        assert value['status']=='INCOMPLETE' and value['actual_counts'] is None and cleaned==['own']
        assert value['reused_old35_outer_actual_counts'] is None
        assert value['finite_suite_task_qualification_passed'] is None


def test_pure_aggregate_keeps_mixed_original_report_labels_and_failures():
    prior=r.old35.pinned('test_completion37_pureaggregate31',ROOT/'scripts/read_b22_suite31.py')
    reports=[]
    for index,stage in enumerate(r.old35.stages()):
        item={'stage':stage['name'],'status':'COMPLETE','actual_counts':{'controls':stage['controls'],
            'normal_native':5*stage['controls'],'compiler_native':2},'schema':'C35-original' if index<11 else 'C37-new'}
        if stage['kind']=='step':item['cases']=[{'actor':'zero','task_qualified':True},{'actor':'final_policy','task_qualified':True}]
        else:item.update(pairing_identity={'same':stage['case_id']},cases=[{'case_id':stage['case_id'],
            'actor':'zero' if stage['actor']=='zero' else 'final_policy','mirror':stage['mirror'],'task_passed':index!=12}])
        reports.append(item)
    before=json.dumps(reports,sort_keys=True)
    scorer_calls=[]
    result=prior.aggregate(reports,lambda original:scorer_calls.append(original) or {'RL_contribution_passed':False})
    assert len(scorer_calls[0])==12 and all(x['mirror'] is False for x in scorer_calls[0])
    assert len(result['mirror_scores_separate'])==len(result['real_15mm_scores_separate'])==2
    assert result['all_seven_B_script_tasks_passed'] is False and result['RL_contribution_passed'] is False
    assert json.dumps(reports,sort_keys=True)==before


def test_new_authored_commit_gate_rejects_uncommitted_blob(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'AUTHORED',('owner.py',))
    path=tmp_path/'owner.py';path.write_bytes(b'current source')
    source={'git_commit':'a'*40,'authored':{str(path):r.c.identity(path)}}
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:b'other committed bytes')
    with pytest.raises(ValueError,match='uncommitted'):r.check_authored(source)
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:b'current source')
    r.check_authored(source)



def test_signal_during_owned_cleanup_is_deferred_and_prevents_next_session(tmp_path):
    import signal
    import time
    previous={n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM)}
    calls=[]
    class Process:
        returncode=0
        def poll(self):return 0
    class Host:
        def clean_descendants(self,parent,process):
            calls.append(process)
            signal.raise_signal(signal.SIGTERM)
            return {'remaining':{}}
    with pytest.raises(InterruptedError,match='termination during owned cleanup'):
        r.supervise(['fake'],tmp_path/'stdout',host=Host(),sample=lambda *a:None,
            limit=10,deadline=time.monotonic()+10,factory=lambda *a,**k:Process())
    assert len(calls)==1 and previous=={n:signal.getsignal(n) for n in previous}


def test_spawn_pre_handle_failure_still_cleans_only_owned_descendants(tmp_path):
    import time
    calls=[]
    class Host:
        def clean_descendants(self,parent,process):calls.append(process);return {'remaining':{}}
    def factory(*a,**k):raise InterruptedError('fake pre-return failure')
    with pytest.raises(InterruptedError,match='pre-return'):
        r.supervise(['fake'],tmp_path/'stdout',host=Host(),sample=lambda *a:None,
            limit=10,deadline=time.monotonic()+10,factory=factory)
    assert calls==[None]
