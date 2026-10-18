"""No real libraries: cold-loader stand-ins, pinned AST and synthetic actions."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('numpy_worker38_test',ROOT/'scripts/run_b22_numpy_worker38.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def test_default_real_cli_plan_is_cold_even_with_an_unrelated_installed_scripts_package():
    code = ('import importlib.util,sys;'
        's=importlib.util.spec_from_file_location("w","scripts/run_b22_numpy_worker38.py");'
        'w=importlib.util.module_from_spec(s);s.loader.exec_module(w);'
        'assert w.main([])==0;'
        'assert not any(n in sys.modules for n in ("numpy","mujoco","torch","stable_baselines3","gymnasium"))')
    result = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr
    value = json.loads(result.stdout)
    assert value['status'] == 'PLAN_ONLY' and value['execution_started'] is False
    assert value['script_controls'] == 3200 and value['normal_native'] == 16000 and value['compiler_native'] == 4
    assert value['numpy_predict_calls'] == 3202 and value['numpy_predict_rows'] == 3264


@pytest.fixture
def cold(tmp_path,monkeypatch):
    paths = {name:tmp_path/(name+'.so') for name in ('library','elf0','elf1','elf2','binding')}
    for name,path in paths.items():path.write_text('synthetic '+name)
    source = {str(path):w.identity(path) for path in paths.values()}
    env = {'LD_PRELOAD':str(paths['library']),'LD_BIND_NOW':'1','LD_LIBRARY_PATH':None,
           'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for key,value in env.items():
        if value is None:monkeypatch.delenv(key,raising=False)
        else:monkeypatch.setenv(key,value)
    monkeypatch.setattr(w.sys,'version_info',(3,10,12))
    monkeypatch.setattr(w.backend,'deny_frameworks38',lambda:w.require(w.backend.frameworks_absent38(),'forbidden imported'))
    monkeypatch.setattr(w.backend,'helper36',lambda:SimpleNamespace(dependency_origins=lambda *a:{'numpy':'synthetic','mujoco':'synthetic'}))
    state = {key:0 for key in w.ZERO_FIELDS}
    state['original_step'] = 12345  # Resolved pointer is not a counter.
    slots = [{'elf':str(paths[name]),'elf_sha256':source[str(paths[name])]['sha256'],
              'passed':True,'target':23456,'expected_wrapper':23456} for name in ('elf0','elf0','elf1','elf2')]
    proof = {'passed':True,'dso_path':env['LD_PRELOAD'],'dso_sha256':source[env['LD_PRELOAD']]['sha256'],
             'jump_slots':slots,'elf_hashes':{str(path):source[str(path)]['sha256'] for name,path in paths.items() if name != 'binding'},
             'counters':dict(state)}
    calls = []
    class Guard:
        def __init__(self,path):calls.append(('guard',path))
        def snapshot(self):calls.append(('snapshot',));return dict(state)
        def binding_proof(self):calls.append(('binding',));return copy.deepcopy(proof)
    binding = SimpleNamespace(__file__=str(paths['binding']),EngineGuard=Guard)
    cache = {'hits':0,'misses':0,'maxsize':1,'currsize':0}
    geometry = SimpleNamespace(cache_info=lambda:SimpleNamespace(_asdict=lambda:dict(cache)))
    monkeypatch.setattr(w,'bootstrap_components38',lambda:(SimpleNamespace(__version__='synthetic'),binding,geometry))
    session = {'library':env['LD_PRELOAD'],'binding_module':str(paths['binding']),
               'source_hashes':source,'runtime_environment':env}
    (tmp_path/'session.json').write_text(json.dumps(session))
    return SimpleNamespace(session=session,output=tmp_path,state=state,proof=proof,cache=cache,calls=calls)


@pytest.mark.parametrize('fault',[None,'native','armed','target','limit','first_caller','bool_counter',
    'got','elf','missing_slot','proof','geometry','preload','ldpath','session','session_symlink'])
def test_actual_cold_bootstrap_gate_requires_zero_counts_and_real_binding_proof(cold,monkeypatch,fault):
    if fault == 'native':cold.state['control_returns'] = 1
    elif fault == 'armed':cold.state['armed'] = 1
    elif fault == 'target':cold.state['target_model'] = 123
    elif fault == 'limit':cold.state['control_limit'] = 16000
    elif fault == 'first_caller':cold.state['first_control_step_caller'] = 99
    elif fault == 'bool_counter':cold.state['ccd_returns'] = False
    elif fault == 'got':cold.proof['jump_slots'][3]['target'] += 1
    elif fault == 'elf':cold.proof['elf_hashes'][cold.session['library']] = '0'*64
    elif fault == 'missing_slot':cold.proof['jump_slots'].pop()
    elif fault == 'proof':cold.proof['passed'] = False
    elif fault == 'geometry':cold.cache['misses'] = 1
    elif fault == 'preload':monkeypatch.setenv('LD_PRELOAD','other.so')
    elif fault == 'ldpath':monkeypatch.setenv('LD_LIBRARY_PATH','')
    elif fault == 'session':(cold.output/'session.json').write_text('{}')
    elif fault == 'session_symlink':
        (cold.output/'session.json').rename(cold.output/'foreign_session.json')
        (cold.output/'session.json').symlink_to(cold.output/'foreign_session.json')
    if fault:
        with pytest.raises(ValueError):w.bootstrap38(cold.session,cold.output)
        assert not (cold.output/'bootstrap38_report.json').exists()
    else:
        value = w.bootstrap38(cold.session,cold.output)
        assert value == json.loads((cold.output/'bootstrap38_report.json').read_text())
        assert value['session_identity'] == w.identity(cold.output/'session.json')
        assert value['actual_counts'] == {'controls':0,'normal_native':0,'compiler_native':0}
        assert value['actual_nn_counts'] == {'predict_calls':0,'predict_rows':0,'ppo_load':0,'torch_load':0}
        assert value['environment_constructions'] == value['numpy_actor_bundle_loads'] == 0
        assert value['state_before']['original_step'] == 12345 and value['qualification_granted'] is False
        assert [row[0] for row in cold.calls] == ['guard','snapshot','binding','snapshot']


def test_cold_bootstrap_actual_source_imports_both_native_extensions_and_never_arms_or_constructs():
    function = next(node for node in ast.parse(Path(w.__file__).read_text()).body
                    if isinstance(node,ast.FunctionDef) and node.name=='bootstrap_components38')
    imports = [name.name for node in ast.walk(function) if isinstance(node,ast.Import) for name in node.names]
    assert imports[:3] == ['mujoco','mujoco._functions','mujoco._rollout']
    function = next(node for node in ast.parse(Path(w.__file__).read_text()).body
                    if isinstance(node,ast.FunctionDef) and node.name=='bootstrap38')
    calls = [ast.unparse(node.func) for node in ast.walk(function) if isinstance(node,ast.Call)]
    assert all(not name.endswith(('.arm','.set_phase','.construct','.reset','.step','.rollout','.MjModel','.MjData',
                                 '.mj_step','.mj_forward','.from_xml_path','.from_xml_string')) for name in calls)
    assert 'guard.binding_proof' in calls and calls.count('guard.snapshot') == 2


def test_full_derived_module_restores_to_original_ast_after_only_explicit_reviewed_seams():
    original,derived = ast.parse(w.ORIGINAL.read_text()),ast.parse(w.derived_source38())
    old_load = next(node for node in ast.walk(original) if isinstance(node,ast.Compare)
                    and ast.unparse(node.left).startswith("counts.get('load'"))
    old_torch = next(node for node in ast.walk(original) if isinstance(node,ast.Compare)
                    and ast.unparse(node.left).startswith("counts.get('torch_load'"))
    class Restore(ast.NodeTransformer):
        def visit_Constant(self,node):
            values = {w.CONTRACT:'C34_B22_residual_gain_probe_v1','gain38':'gain34',
                'b22-numpy38-worker-receipt-v1':'d1-c23-gui-worker-receipt-v1',
                'b22_numpy_backend38':'rl16_learning_11','probe_actions_within_tolerance':'probe_actions_byte_exact'}
            if isinstance(node.value,str) and node.value in values:node.value = values[node.value]
            return node
        def visit_Name(self,node):
            if node.id == 'load_and_verify_numpy38':node.id = 'load_and_verify_final'
            return node
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_bootstrap38':
                node.func = ast.Attribute(value=ast.Name(id='common',ctx=ast.Load()),attr='_load_import_repair',ctx=ast.Load())
            return self.generic_visit(node)
        def visit_Compare(self,node):
            if ast.unparse(node.left).startswith("counts.get('load'"):return copy.deepcopy(old_load)
            if ast.unparse(node.left).startswith("counts.get('torch_load'"):return copy.deepcopy(old_torch)
            if ast.unparse(node.left)=="session['gain38']":
                node.ops,node.comparators = [ast.NotIn()],[ast.Tuple(elts=[ast.Constant(1.),ast.Constant(.5)],ctx=ast.Load())]
            return self.generic_visit(node)
        def visit_ImportFrom(self,node):
            if node.module == 'b22_numpy_backend38':
                alias = node.names[0]
                if alias.name == 'audited_origins38':return None
                if alias.name == 'NumpyCounter38':node.module,node.names = 'runtime_support22',[ast.alias(name='ModelCounter')]
                elif alias.name == 'load_and_verify_numpy38':node.module,node.names = 'rl16_learning_11',[ast.alias(name='load_and_verify_final')]
            elif node.module == 'run_rl16_training_08' and not any(a.name=='_audited_learning_origins' for a in node.names):
                node.names.append(ast.alias(name='_audited_learning_origins'))
            return node
    restored = Restore().visit(derived)
    restored.body[0] = copy.deepcopy(original.body[0])  # versioned module docstring
    assert ast.dump(restored,include_attributes=False) == ast.dump(original,include_attributes=False)


def test_actual_derived_control_block_applies_half_exactly_once_and_logs_network_output():
    tree = ast.parse(w.derived_source38())
    assignments = next(node.body for node in ast.walk(tree) if isinstance(node,ast.While)
                       and any(isinstance(item,ast.Assign) and ast.unparse(item.targets[0])=='(network_output16, predicted)' for item in node.body))
    begin = next(i for i,node in enumerate(assignments) if isinstance(node,ast.Assign)
                 and ast.unparse(node.targets[0])=='(network_output16, predicted)')
    block = copy.deepcopy(assignments[begin:begin+3])
    network = np.linspace(-1.,1.,16,dtype=np.float32)
    namespace = {'np':np,'session':{'gain38':.5},'policy':object(),'observation':'synthetic',
                 '_numeric_actor_action':lambda *a:(network.copy(),True)}
    exec(compile(ast.Module(body=block,type_ignores=[]),'pinned_C38_gain_block','exec'),namespace)  # noqa: S102 -- three exact reviewed assignments, synthetic action
    assert np.array_equal(namespace['network_output16'],network)
    assert np.array_equal(namespace['action'],network*np.float32(.5))
