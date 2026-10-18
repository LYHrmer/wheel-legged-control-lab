"""C39 pure AST/metadata/fake lifecycle regressions; zero NN and physics."""
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

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('test_b22_numpy39_driver',ROOT/'scripts/run_b22_numpy39.py')
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


@pytest.fixture
def graph(monkeypatch):
    package = SimpleNamespace(inventory39=lambda:{'schema':'synthetic-package39','inputs':{}},
        validate_inventory39=lambda saved:None,verify_loaded_roots39=lambda saved:None)
    monkeypatch.setattr(d,'package_helper',lambda:package)
    return d.build_scope()


def test_inverse_ast_exact_and_every_old_sha_retained(graph):
    assert len(graph.proof['legacy_inputs']) == 8
    for kind,name in d.MODULES.items():
        tree,proof = d.derive(kind)
        assert proof == graph.proof['modules'][kind]
        assert proof['inverse_ast_exact'] and proof['all_sha_constants_unchanged']
        assert proof['original_sha256'] == d.LEGACY[name]
        assert ast.dump(tree,include_attributes=False)
    assert graph.runner.AUTHORED == d.AUTHORED
    assert len(d.AUTHORED) == 5
    assert graph.runner.CONTRACT == graph.worker.CONTRACT == graph.backend.CONTRACT == d.CONTRACT
    assert graph.runner.OUTPUT == d.OUTPUT


def test_unbound_legacy_source_is_rejected_before_module_execution(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'ROOT',tmp_path)
    for name in d.LEGACY:
        p = tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('tamper')
    with pytest.raises(ValueError,match='frozen C38 source changed'):
        d.build_scope()


def test_identity_seams_never_replace_arbitrary_version_digits_or_shas():
    for digest in d.LEGACY.values():assert d.identity_literal(digest) == digest
    assert d.identity_literal('12438.0') == '12438.0'
    assert d.identity_literal('results/b22_numpy36_20261018_01') == 'results/b22_numpy36_20261018_01'
    assert d.identity_literal('C35_B22_half_gain_fixed_suite_v1') == 'C35_B22_half_gain_fixed_suite_v1'
    assert d.identity_literal("'schema':'b22-numpy38-worker-receipt-v1'") == "'schema':'b22-numpy39-worker-receipt-v1'"


def test_injected_modules_have_real_new_source_and_one_counter_type(graph,monkeypatch):
    assert graph.runner.backend is graph.worker.backend is graph.backend
    assert graph.reader.r is graph.runner
    assert sys.modules['b22_numpy_backend39'] is graph.backend
    for module in (graph.runner,graph.worker,graph.backend,graph.reader):
        assert module.__file__ == str(d.DRIVER)
    monkeypatch.setattr(graph.backend,'frameworks_absent38',lambda:True)
    with graph.backend.NumpyCounter38(False,graph.backend.API_LIMITS) as counter:
        assert graph.backend._LOCAL.counter is counter
        counter.phase = 'probe';assert counter.call(32,lambda:'fake') == 'fake'
        counter.phase = 'control';assert counter.call(1,lambda:'fake') == 'fake'
        value = counter.summary()
        assert value['schema'] == 'b22-numpy39-model-api-ledger-v1'
        assert value['counts']['predict']['returned'] == 2
        assert value['counts']['predict']['rows_returned'] == 33
    assert graph.backend._LOCAL.counter is None
    source = graph.worker.derived_source38()
    assert 'from b22_numpy_backend39 import NumpyCounter38 as ModelCounter' in source
    assert 'from b22_numpy_backend38' not in source
    assert "'schema':'b22-numpy39-worker-receipt-v1'" in source
    assert graph.worker.derivation_proof38(source)['execution_contract_id'] == d.CONTRACT


@pytest.mark.parametrize('argv,owner',[
    (['--session','/session','--output','/run'],'worker'),
    (['--bootstrap','--session','/session','--output','/run'],'worker'),
    (['--release','/release','--stage','flat_1p6_B','--report','/report'],'reader'),
    (['--release','/release','--compare','--report','/report'],'reader')])
def test_exact_host_argv_without_worker_flag_and_both_read_modes(monkeypatch,argv,owner):
    calls = []
    graph = SimpleNamespace(worker=SimpleNamespace(main=lambda value:calls.append(('worker',value)) or 0),
        reader=SimpleNamespace(main=lambda value:calls.append(('reader',value)) or 0))
    monkeypatch.setattr(d,'scope',lambda:graph)
    assert d.main(argv) == 0 and calls == [(owner,argv)]


@pytest.mark.parametrize('argv',[
    ['--session','/s','--execute'],['--session','/s','--release','/r'],
    ['--release','/r','--execute'],['--release','/r','--bootstrap']])
def test_mixed_parent_worker_reader_routes_rejected(monkeypatch,argv):
    monkeypatch.setattr(d,'scope',lambda:SimpleNamespace())
    with pytest.raises(ValueError,match='mixed'):d.main(argv)


def test_plan_default_uses_same_budget_without_execution(graph,monkeypatch,capsys):
    monkeypatch.setattr(d,'scope',lambda:graph)
    monkeypatch.setattr(graph.runner,'execute',lambda *_:pytest.fail('executed'))
    assert d.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['schema'] == 'b22-numpy39-plan-v1' and value['status'] == 'PLAN_ONLY'
    assert value['output'] == str(d.OUTPUT) and not value['execution_started']
    assert value['stage_order'] == ['bootstrap','flat_1p6_B','yaw_1p2_B']
    assert value['limits']['controls'] == 3200 and value['limits']['normal_native'] == 16000
    assert value['limits']['compiler_native'] == 4
    assert value['limits']['numpy_predict_calls'] == 3202 and value['limits']['numpy_predict_rows'] == 3264
    assert value['limits']['start_effective_bytes'] == 6 << 30
    assert value['limits']['reserve_effective_bytes'] == 4 << 30
    assert value['limits']['admission_wait_s'] == 300 and value['limits']['total_wall_s'] == 900
    assert not value['retry_permitted'] and not value['qualification_granted']
    assert value['runtime_package_inventory'] == graph.package.inventory39()


def test_merge_rejects_package_identity_collision():
    a = {'/fixed':{'bytes':1,'sha256':'a'*64}}
    assert d.merge_inputs(a,a) == a
    with pytest.raises(ValueError,match='overlapping'):
        d.merge_inputs(a,{'/fixed':{'bytes':1,'sha256':'b'*64}})


@pytest.fixture
def release_fixture(graph,monkeypatch,tmp_path):
    r = graph.runner
    monkeypatch.setattr(r,'ROOT',tmp_path)
    output = tmp_path/'results/new39';output.mkdir(parents=True)
    monkeypatch.setattr(r,'OUTPUT',output)
    cfg = {'source_hashes':{},'shared_session':{}}
    config_path = tmp_path/r.c.MANIFEST;config_path.parent.mkdir(parents=True,exist_ok=True)
    r.c.write_json(config_path,cfg)
    references = {'inputs':{},'old_inputs':{}}
    monkeypatch.setattr(r,'reference35',lambda:references)
    monkeypatch.setattr(r.backend,'export_identities38',dict)
    monkeypatch.setattr(r,'check_authored',lambda _:None)
    source = {'authored':{}}
    (tmp_path/'scripts').mkdir(exist_ok=True)
    (tmp_path/'scripts/run_b22_gain_suite35.py').write_text('# fake closure only')
    release = {'schema':'b22-numpy39-release-v1','plan':r.plan(),'source':source,
        'references35':{},'export36':{},'inputs':r.closure(source,references,{},cfg),'v27_readiness':{},
        'runtime_package_inventory':graph.package.inventory39(),'derivation39':copy.deepcopy(graph.proof)}
    path = output/'release.json';r.c.write_json(path,release)
    return graph,path,release


def test_release_exact_new_closure_and_derivation_accepted(release_fixture):
    graph,path,release = release_fixture
    assert graph.runner.validate_release(path) == release
    assert all(name in release['inputs'] for name in graph.proof['legacy_inputs'])


@pytest.mark.parametrize('fault',['extra','legacy_missing','old_schema','old_output','proof','inventory'])
def test_release_never_accepts_partial_or_relabelled_closure(release_fixture,fault):
    graph,path,release = release_fixture
    if fault == 'extra':release['extra'] = True
    elif fault == 'legacy_missing':release['inputs'].pop(next(iter(graph.proof['legacy_inputs'])))
    elif fault == 'old_schema':release['schema'] = 'b22-numpy38-release-v1'
    elif fault == 'old_output':release['plan']['output'] = 'results/b22_numpy38_20261018_01'
    elif fault == 'proof':release['derivation39']['modules']['worker']['original_sha256'] = '0'*64
    else:
        release['runtime_package_inventory']['bad'] = True
        graph.package.validate_inventory39 = lambda saved: (_ for _ in ()).throw(ValueError('inventory'))
    graph.runner.atomic_json(path,release)
    with pytest.raises(ValueError):graph.runner.validate_release(path)


def test_actual_new_parent_argv_and_runtime_origin_mapping(graph):
    r = graph.runner
    argv = r.bootstrap_argv(d.OUTPUT/'bootstrap/session.json',d.OUTPUT/'bootstrap')
    assert argv == ['/usr/bin/python3','-B',str(d.DRIVER),'--bootstrap','--session',
                    str(d.OUTPUT/'bootstrap/session.json'),'--output',str(d.OUTPUT/'bootstrap')]
    tree,_ = d.derive('reader')
    strings = [n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert any("expected_origins['b22_numpy_backend39'] = str(c.ROOT/'scripts/run_b22_numpy39.py')" in x for x in strings)
    assert not any('scripts/run_b22_numpy_worker38.py' in x for x in strings)


@pytest.fixture
def execution(graph,monkeypatch,tmp_path):
    # Reuse the pinned C38 synthetic process fixture and its inverse-locked
    # lifecycle assertions. Only new schema/path strings and module injection
    # differ; this fixture never invokes any real Popen or actor.
    source = (ROOT/'tests/test_b22_numpy38.py').read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):node.value = d.identity_literal(node.value)
    module = SimpleNamespace(__file__=str(ROOT/'tests/test_b22_numpy39.py'),__name__='fake39_legacy_tests')
    # Import-time loading remains pure; injected modules are assigned after it.
    # The original load calls would target driver39 rather than a runner, so
    # omit those two assignments and supply the already-derived objects.
    tree.body = [n for n in tree.body if not d._target(n,'r') and not d._target(n,'d')]
    module.__dict__.update(r=graph.runner,d=graph.reader)
    exec(compile(ast.fix_missing_locations(tree),module.__file__,'exec'),module.__dict__)  # noqa: S102 -- pinned synthetic fixture only
    state = module.execution.__wrapped__(tmp_path,monkeypatch)
    monkeypatch.setattr(graph.runner,'_scope_reader',lambda:SimpleNamespace(verify_bootstrap=lambda *_:True))
    return module,state


def test_full_fake_lifecycle_reuses_original_supervision_and_no_retry(execution):
    module,state = execution
    module.test_fake_lifecycle_cold_bootstrap_then_two_numpy_physical_cases_and_saved_readback(state)


@pytest.mark.parametrize('fault',['bootstrap','flat_1p6_B','flat_1p6_B_reader','yaw_1p2_B','comparison','resource','late'])
def test_fake_failures_keep_original_unknown_and_owned_cleanup(execution,fault):
    module,state = execution
    module.test_engineering_failure_never_retries_and_unknown_actual_is_not_zero(state,fault)


def test_fake_low_initial_ram_never_claims(execution):
    module,state = execution
    module.test_initial_admission_below6_does_not_claim_output_or_spawn(state)


def test_fake_ram_recovery_wait_keeps_one_spawn(execution,monkeypatch):
    module,state = execution
    module.test_no_live_child_waits_for6_before_creating_next_request(state,monkeypatch)


@pytest.mark.parametrize('fault',['timeout','below4','unknown','signal','remaining_window','late_source'])
def test_fake_ram_wait_failures_keep_original_windows(execution,monkeypatch,fault):
    module,state = execution
    module.test_admission_wait_stops_without_claim_or_retry(state,monkeypatch,fault)


def test_cleanup_signal_capture_preserved(graph):
    def cleanup(parent,process):
        signal.raise_signal(signal.SIGTERM)
        return {'remaining':{}}
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError,match='termination during owned cleanup'):
        graph.runner.cleanup_owned(SimpleNamespace(clean_descendants=cleanup),None)
    assert signal.getsignal(signal.SIGTERM) == before


@pytest.mark.parametrize('bound',[False,True])
def test_original_actual_origin_gate_rejects_unbound_enums_and_accepts_expanded_map(release_fixture,monkeypatch,tmp_path,bound):
    graph,path,_release = release_fixture
    # Execute the original dependency_origins function with synthetic modules;
    # no Python numeric package is imported and no actor is evaluated.
    source = (ROOT/'scripts/export_b22_numpy36.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'dependency_origins')
    modules,sources = {},{}
    for name in ('numpy','gymnasium','mujoco','mujoco._enums'):
        item = tmp_path/(name+'.py');item.write_text('# fake '+name)
        modules[name] = SimpleNamespace(__file__=str(item))
        if bound or name != 'mujoco._enums':sources[str(item)] = d.identity(item)
    namespace = {'Path':Path,'sys':SimpleNamespace(modules=modules),'require':d.require,'identity':d.identity}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned pure origins>','exec'),namespace)  # noqa: S102 -- original pure checker with fake modules
    monkeypatch.setattr(graph.backend,'helper36',lambda:SimpleNamespace(dependency_origins=namespace['dependency_origins']))
    monkeypatch.setattr(graph.backend,'frameworks_absent38',lambda:True)
    session = {'source_hashes':{str(path):d.identity(path),**sources}}
    if bound:
        value = graph.backend.audited_origins38(session)
        assert value['schema'] == 'b22-numpy39-runtime-origins-v1'
        assert value['module_origins']['mujoco._enums'] == str(tmp_path/'mujoco._enums.py')
    else:
        with pytest.raises(ValueError,match='unbound numeric import: mujoco._enums'):
            graph.backend.audited_origins38(session)
