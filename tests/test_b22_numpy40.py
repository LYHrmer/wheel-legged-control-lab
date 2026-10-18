"""C40 saved-only identity/seam/lifecycle tests; no runtime package or physics."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('test_numpy40',ROOT/'scripts/read_b22_numpy40.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def original_text():
    path = ROOT/'scripts/read_b22_suite31.py'
    text = path.read_text()
    return ast.get_source_segment(text,next(n for n in ast.parse(text).body
        if isinstance(n,ast.FunctionDef) and n.name == 'script_sources'))


def map_loop(text):
    return next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.For)
        and isinstance(n.iter,ast.Tuple)
        and [getattr(x,'value',None) for x in n.iter.elts] == list(m.MAP_FILES))


def run_loop(node,documents):
    namespace = {'base':SimpleNamespace(saved_document=lambda path:documents[path.name]),
        'run':Path('/saved'),'sources':{'/bound/torch/lib.so':{}},'c':SimpleNamespace(require=m.require)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),  # noqa: S102 -- exact tracked AST, synthetic documents
        'pinned_torch_map_loop40','exec'),namespace)


def test_actual_worker22_selects_torch_nvidia_only_and_old_reader_rejects_legal_empty(tmp_path):
    source = ast.parse((ROOT/'runtime/d1_b22/research/continuation22/worker22.py').read_text())
    node = next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name == 'mapped_libraries')
    maps = tmp_path/'maps'
    maps.write_text('1000-2000 r-xp 0000 08:01 1 /lib/libc.so.6\n'
        '2000-3000 r-xp 0000 08:01 2 /site/mujoco/_functions.so\n'
        '3000-4000 r-xp 0000 08:01 3 /site/numpy.libs/libopenblas.so\n')
    namespace = {'Path':lambda path:maps if str(path) == '/proc/self/maps' else Path(path)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_mapped_libraries22','exec'),namespace)  # noqa: S102 -- real small proc-map parser, no module import
    assert namespace['mapped_libraries']({}) == {'paths':[]}
    with pytest.raises(ValueError,match='actual DSO left closure'):
        run_loop(map_loop(original_text()),{name:{'paths':[]} for name in m.MAP_FILES})


@pytest.mark.parametrize('document',[{'paths':[]},{'paths':['/bound/torch/lib.so']},
    {'paths':['/unbound.so']},{'paths':[],'extra':True},{},{'paths':None},[],{'paths':False}])
def test_new_gate_requires_exact_empty_even_if_nonempty_path_was_bound(document):
    documents = {name:copy.deepcopy(document) for name in m.MAP_FILES}
    if document == {'paths':[]}:
        run_loop(map_loop(m.adapt_maps40(original_text())),documents)
    else:
        with pytest.raises(ValueError,match='exactly empty'):
            run_loop(map_loop(m.adapt_maps40(original_text())),documents)


def test_only_one_loop_body_changes_and_every_other_metadata_ast_is_identical():
    before = ast.parse(original_text())
    after = ast.parse(m.adapt_maps40(original_text()))
    old_loop,new_loop = map_loop(ast.unparse(before)),map_loop(ast.unparse(after))
    actual_new = next(n for n in ast.walk(after) if isinstance(n,ast.For)
        and ast.dump(n.iter) == ast.dump(new_loop.iter))
    actual_new.body = copy.deepcopy(old_loop.body)
    assert ast.dump(before,include_attributes=False) == ast.dump(after,include_attributes=False)


@pytest.mark.parametrize('fault',['missing','guard_changed','duplicate'])
def test_unexpected_original_seam_is_rejected(fault):
    text = original_text()
    if fault == 'missing':text = text.replace('mapped_libraries_final.json','other.json')
    elif fault == 'guard_changed':text = text.replace('actual DSO left closure','another error')
    else:text += "\nfor filename in ('mapped_libraries_before_load.json','mapped_libraries_final.json'):\n    pass\n"
    with pytest.raises(ValueError):m.adapt_maps40(text)


def test_actual39_adapter_keeps_all_other_gates_and_emits_the_new_exact_loop(monkeypatch):
    driver,graph = m.load_graph39()
    names = ('verify_bootstrap','verify_origins','verify_numpy_ledger','verify_load_proof','check_gain_row','read_stage')
    unchanged = {name:getattr(graph.reader,name) for name in names}
    completion = graph.runner.case_completion
    proof = m.install_adapter40(driver,graph)
    assert proof['inverse_ast_exact'] and graph.runner.case_completion is completion
    assert all(getattr(graph.reader,name) is function for name,function in unchanged.items())
    seen = []
    def adapt(text):
        seen.append(text)
        return m.adapt_maps40(text)
    monkeypatch.setitem(graph.reader.metadata_checker.__globals__,'_adapt_maps40',adapt)
    prior = graph.runner.old35.pinned('test40_metadata31',ROOT/'scripts/read_b22_suite31.py')
    check = graph.reader.metadata_checker(prior)
    assert check.__name__ == 'script_sources' and len(seen) == 1
    run_loop(map_loop(m.adapt_maps40(seen[0])),{name:{'paths':[]} for name in m.MAP_FILES})


def test_tree_identity_detects_content_addition_removal_and_rejects_symlink(tmp_path):
    folder = tmp_path/'original39';folder.mkdir()
    p = folder/'payload';p.write_bytes(b'old')
    before = m.tree_identity(folder)
    p.write_bytes(b'new')
    assert m.tree_identity(folder) != before
    p.write_bytes(b'old')
    (folder/'extra').write_bytes(b'new')
    assert m.tree_identity(folder) != before
    (folder/'extra').unlink()
    p.rename(folder/'renamed')
    assert m.tree_identity(folder) != before
    (folder/'symlink').symlink_to(folder/'renamed')
    with pytest.raises(ValueError,match='symlink'):m.tree_identity(folder)


def test_default_real_cli_is_cold_and_never_creates_output():
    code = ('import importlib.util,sys,json,contextlib,io;'
        's=importlib.util.spec_from_file_location("r","scripts/read_b22_numpy40.py");'
        'r=importlib.util.module_from_spec(s);s.loader.exec_module(r);out=io.StringIO();'
        '\nwith contextlib.redirect_stdout(out):assert r.main([])==0\n'
        'v=json.loads(out.getvalue());assert v["status"]=="PLAN_ONLY";'
        'assert not any(n in sys.modules for n in ("numpy","mujoco","torch","stable_baselines3","gymnasium"));'
        'assert not v["execution_started"] and v["new_actual_counts"]==r.ZERO')
    value = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=False)
    assert value.returncode == 0,value.stderr


@pytest.mark.parametrize('name',sorted(m.FORBIDDEN))
def test_real_cold_gate_rejects_preloaded_sentinel_without_importing_runtime(name):
    code = ('import importlib.util,sys,types;'
        's=importlib.util.spec_from_file_location("r","scripts/read_b22_numpy40.py");'
        'r=importlib.util.module_from_spec(s);s.loader.exec_module(r);'
        f'sentinel=types.ModuleType({name!r});sys.modules[{name!r}]=sentinel;'
        '\ntry:r.deny_runtime40()\nexcept ValueError:pass\nelse:raise AssertionError("accepted runtime")\n'
        f'assert sys.modules[{name!r}] is sentinel')
    value = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=False)
    assert value.returncode == 0,value.stderr


@pytest.fixture
def execution(tmp_path,monkeypatch):
    old,new = tmp_path/'original39',tmp_path/'new40'
    old.mkdir();payload = old/'payload';payload.write_text('closed old archive')
    monkeypatch.setattr(m,'ORIGINAL',old);monkeypatch.setattr(m,'OUTPUT',new)
    monkeypatch.setattr(m,'deny_runtime40',lambda:None)  # Pure lifecycle boundary; real cold gate is isolated above.
    monkeypatch.setattr(m.os,'getpriority',lambda *args:10)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'1')
    source = {'git_commit':'a'*40,'authored':{}}
    state = SimpleNamespace(now=0.,ram=8 << 30,calls=0,fault=None)
    saved = {'actual_counts':dict(m.EXPECTED_COUNTS),'numpy_predict_calls':1601,'numpy_predict_rows':1632,
        'ppo_load':0,'torch_load':0,'reader_physics_calls':0,'reader_nn_forward_calls':0,
        'cases':[{'task_passed':False}],'pairing_identity':{'reset':'same'},'reader_identity':{'sha256':'b'*64},
        'artifacts':{'payload':m.identity(payload)}}
    def read_stage(path,stage):
        state.calls += 1
        if state.fault == 'reader':raise ValueError('synthetic saved-record error')
        if state.fault == 'mutate':payload.write_text('changed old archive')
        if state.fault == 'count':saved['actual_counts']['controls'] = 1599
        if state.fault == 'nn':saved['numpy_predict_calls'] = 1600
        if state.fault == 'reader_nn':saved['reader_nn_forward_calls'] = 1
        if state.fault == 'ram':state.ram = (4 << 30)-1
        if state.fault == 'wall':state.now = 601.
        if state.fault == 'signal':signal.raise_signal(signal.SIGTERM)
        return copy.deepcopy(saved)
    graph = SimpleNamespace(runner=SimpleNamespace(supervision=SimpleNamespace(memory=lambda:state.ram),
        validate_release=lambda path:None),reader=SimpleNamespace(read_stage=read_stage))
    monkeypatch.setattr(m,'install_adapter40',lambda *args:{'inverse_ast_exact':True})
    monkeypatch.setattr(m,'check_original39',lambda:({'source':{'git_commit':m.SOURCE_COMMIT39},'inputs':{}},{}))
    state.run = lambda:m.execute(graph_loader=lambda:(None,graph),source_reader=lambda:source,clock=lambda:state.now)
    state.new,state.old = new,old
    return state


def test_fake_readback_has_new_identity_keeps_original_counts_unknown_and_does_not_retry(execution):
    before = m.tree_identity(execution.old)
    handlers = {s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    report = execution.run()
    assert report['status'] == 'COMPLETE' and report['new_actual_counts'] == m.ZERO
    assert report['original39_parent_status'] == 'INCOMPLETE'
    assert report['original39_actual_counts'] is report['original39_actual_nn_counts'] is None
    assert report['verified_saved_execution_counts'] == m.EXPECTED_COUNTS
    assert report['verified_saved_nn_counts']['predict_calls'] == 1601
    assert not report['cases'][0]['task_passed'] and not report['qualification_granted'] and not report['yaw_evaluated']
    assert m.tree_identity(execution.old) == before and execution.calls == 1
    assert json.loads((execution.new/'report.json').read_text()) == report
    assert all(signal.getsignal(s) == handler for s,handler in handlers.items())
    with pytest.raises(ValueError,match='no retry'):execution.run()


@pytest.mark.parametrize('fault',['reader','mutate','count','nn','reader_nn','ram','wall','signal'])
def test_failure_stays_in_new_output_and_never_fabricates_a_completed_old_run(execution,fault):
    execution.fault = fault
    report = execution.run()
    assert report['status'] == 'INCOMPLETE' and report['new_actual_counts'] == m.ZERO
    assert report['verified_saved_execution_counts'] is report['verified_saved_nn_counts'] is None
    assert report['original39_actual_counts'] is report['original39_actual_nn_counts'] is None
    assert report['failure'] and execution.calls == 1


def test_low_start_ram_does_not_claim_saved_reader_output(execution):
    execution.ram = (6 << 30)-1
    with pytest.raises(ValueError,match='start RAM'):execution.run()
    assert not execution.new.exists() and execution.calls == 0
