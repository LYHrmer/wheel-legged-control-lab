"""Synthetic NumPy arrays and fake artifacts only, never a real actor load."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('backend38_test',ROOT/'scripts/b22_numpy_backend38.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
h = b.helper36()


@pytest.fixture
def synthetic_framework_scope(monkeypatch):
    # Counter/arithmetic fixtures share pytest's collector, which may import
    # Torch elsewhere. Real cold admission is checked in fresh processes below.
    monkeypatch.setattr(b,'frameworks_absent38',lambda:True)


def weights():
    value = {key:np.zeros(shape,np.float32) for key,shape in h.WEIGHTS.items()}
    value['b3'][:] = np.linspace(-2.,2.,16,dtype=np.float32)
    return value


def test_new_no_framework_counter_covers_single32_probe_plus1600_batch1_exactly(synthetic_framework_scope):
    counter = b.NumpyCounter38(False,dict(b.API_LIMITS))
    with counter:
        counter.phase = 'probe'
        counter.call(32,lambda:None)
        with pytest.raises(ValueError,match='cap'):counter.call(32,lambda:None)
        counter.phase = 'control'
        for _ in range(1600):counter.call(1,lambda:None)
        with pytest.raises(ValueError,match='cap'):counter.call(1,lambda:None)
    value = counter.summary()
    assert value['counts']['predict']['returned'] == 1601 and value['actor_rows'] == 1632
    assert value['critic_rows'] == value['ppo_load_calls'] == value['torch_load_calls'] == 0
    assert value['schema'] == 'b22-numpy38-model-api-ledger-v1' and value['backend'] == 'NumPy'


def test_failed_numpy_operation_retains_attempt_without_fabricated_return(synthetic_framework_scope):
    counter = b.NumpyCounter38(False,dict(b.API_LIMITS))
    with counter:
        counter.phase = 'probe'
        def fail():raise OSError('synthetic numerical failure')
        with pytest.raises(OSError):counter.call(32,fail)
    value = counter.summary()['counts']['predict']
    assert value['attempted'] == 1 and value['returned'] == 0 and value['rows_attempted'] == 32 and value['rows_returned'] == 0


@pytest.mark.parametrize('phase,rows',[('setup',1),('probe',1),('control',32),('control',2)])
def test_wrong_phase_or_batch_never_computes(synthetic_framework_scope,phase,rows):
    counter = b.NumpyCounter38(False,dict(b.API_LIMITS))
    called = []
    with counter:
        counter.phase = phase
        with pytest.raises(ValueError):counter.call(rows,lambda:called.append(True))
    assert called == []


@pytest.mark.parametrize('fault',[None,'shape','dtype','nan','outside','stateful','unowned'])
def test_actual_numpy_actor_returns_clip16_without_half_gain_or_input_mutation(synthetic_framework_scope,fault):
    counter = b.NumpyCounter38(False,dict(b.API_LIMITS))
    actor = b.NumpyActor38(weights(),np,counter)
    obs = np.zeros(99,np.float32)
    deterministic = True
    if fault == 'shape':obs = np.zeros((1,99),np.float32)
    elif fault == 'dtype':obs = obs.astype(np.float64)
    elif fault == 'nan':obs[0] = np.nan
    elif fault == 'outside':obs[0] = 5.1
    elif fault == 'stateful':deterministic = False
    with counter:
        counter.phase = 'control'
        if fault == 'unowned':b._LOCAL.counter = None
        if fault:
            with pytest.raises(ValueError):actor.predict(obs,deterministic=deterministic)
        else:
            before = obs.tobytes()
            action,state = actor.predict(obs,deterministic=True)
            assert np.array_equal(action,np.clip(weights()['b3'],-1.,1.))
            assert action.dtype == np.float32 and state is None and obs.tobytes() == before
            assert np.max(abs(action)) == 1.  # A duplicate half gain would fail this.


@pytest.mark.parametrize('fault',['missing','extra','shape','dtype','nan'])
def test_actor_requires_all_six_exact_finite_float32_tensors(fault):
    value = weights()
    if fault == 'missing':value.pop('w3')
    elif fault == 'extra':value['critic'] = np.zeros(1,np.float32)
    elif fault == 'shape':value['w1'] = value['w1'].T
    elif fault == 'dtype':value['b2'] = value['b2'].astype(np.float64)
    elif fault == 'nan':value['b2'][0] = np.nan
    with pytest.raises(ValueError):b.NumpyActor38(value,np,b.NumpyCounter38(False,dict(b.API_LIMITS)))


@pytest.fixture
def bundle(tmp_path,monkeypatch,synthetic_framework_scope):
    folder = tmp_path/'runtime/d1_b22/checkpoint'
    folder.mkdir(parents=True)
    export = tmp_path/'export36';export.mkdir()
    np.savez(export/'actor.npz',**weights())
    obs = np.zeros((32,99),np.float32)
    expected = np.tile(np.clip(weights()['b3'],-1.,1.),(32,1))
    (folder/'final_probe_observations_f32.bin').write_bytes(obs.tobytes())
    (folder/'final_probe_actions_f32.bin').write_bytes(expected.tobytes())
    (folder/'final_model.zip').write_bytes(b'synthetic payload, not a model')
    files = {p.name:h.identity(p) for p in folder.iterdir()}
    h.write_json(export/'actor_manifest.json',{'original_probe_files':files,'versions':{'numpy':np.__version__}})
    monkeypatch.setattr(b,'ROOT',tmp_path)
    monkeypatch.setattr(b,'EXPORT',export)
    monkeypatch.setattr(b,'helper36',lambda:h)
    bound = {str(p):h.identity(p) for p in export.iterdir()}
    monkeypatch.setattr(b,'export_identities38',lambda:dict(bound))
    return SimpleNamespace(folder=folder,export=export,manifest={'files':files,'metadata_sha256':h.METADATA_SHA},expected=expected)


@pytest.mark.parametrize('fault',[None,'metadata','probe','numpy_version','second_load'])
def test_actual_new_loader_uses_synthetic_bundle_and_exact_one32_numeric_probe(bundle,fault):
    manifest = bundle.manifest
    if fault == 'metadata':manifest['metadata_sha256'] = '0'*64
    elif fault == 'probe':(bundle.folder/'final_probe_observations_f32.bin').write_bytes(b'changed')
    elif fault == 'numpy_version':
        value = h.read_json(bundle.export/'actor_manifest.json');value['versions']['numpy'] = 'unknown'
        (bundle.export/'actor_manifest.json').write_text(json.dumps(value))
    counter = b.NumpyCounter38(False,dict(b.API_LIMITS))
    with counter:
        counter.phase = 'probe'
        if fault in ('metadata','probe','numpy_version'):
            with pytest.raises(ValueError):b.load_and_verify_numpy38(bundle.folder,manifest,return_model=True)
        else:
            model,proof = b.load_and_verify_numpy38(bundle.folder,manifest,return_model=True)
            assert proof['schema'] == 'b22-numpy38-load-proof-v1' and proof['probe_actions_within_tolerance'] is True
            assert proof['probe_prediction_calls'] == 1 and proof['probe_rows'] == 32
            assert proof['ppo_load_calls'] == proof['torch_load_calls'] == 0
            assert counter.summary()['counts']['predict']['rows_returned'] == 32
            if fault == 'second_load':
                with pytest.raises(ValueError,match='exactly one'):b.load_and_verify_numpy38(bundle.folder,manifest,return_model=True)
            else:
                counter.phase = 'control'
                action,state = model.predict(np.zeros(99,np.float32))
                assert state is None and np.array_equal(action,bundle.expected[0])


def test_raw_probe_limit_cannot_be_relaxed_because_half_gain_is_small(bundle,monkeypatch):
    original = b.NumpyActor38.predict
    def altered(self,*args,**kwargs):
        action,state = original(self,*args,**kwargs)
        action = action.copy();action[0,8] += np.float32(1.3e-6)
        return action,state
    monkeypatch.setattr(b.NumpyActor38,'predict',altered)
    counter = b.NumpyCounter38(False,dict(b.API_LIMITS))
    with counter:
        counter.phase = 'probe'
        with pytest.raises(ValueError,match='tolerance'):b.load_and_verify_numpy38(bundle.folder,bundle.manifest,return_model=True)
    assert counter.summary()['actor_rows'] == 32


@pytest.mark.parametrize('module',['torch','torch.nn','stable_baselines3','cv2','jax','tensorflow'])
def test_fresh_process_import_guard_blocks_framework_before_execution(module):
    code = ('import importlib.util,sys;'
        's=importlib.util.spec_from_file_location("b","scripts/b22_numpy_backend38.py");'
        'b=importlib.util.module_from_spec(s);s.loader.exec_module(b);b.deny_frameworks38();'
        f'\ntry:__import__({module!r})\nexcept ImportError as e:assert "C38 forbidden" in str(e)\nelse:raise AssertionError("unblocked")\n'
        'assert b.frameworks_absent38() and "numpy" not in sys.modules and "mujoco" not in sys.modules')
    result = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr


@pytest.mark.parametrize('module',['torch','torch.nn','stable_baselines3','cv2','jax','tensorflow'])
def test_fresh_process_real_cold_gates_reject_preloaded_framework_sentinel(module):
    code = ('import importlib.util,sys,types;'
        's=importlib.util.spec_from_file_location("b","scripts/b22_numpy_backend38.py");'
        'b=importlib.util.module_from_spec(s);s.loader.exec_module(b);'
        f'sentinel=types.ModuleType({module!r});sys.modules[{module!r}]=sentinel;'
        'assert not b.frameworks_absent38();'
        '\nfor gate in (b.deny_frameworks38,b.NumpyCounter38(False,dict(b.API_LIMITS)).__enter__):'
        '\n try:gate()\n except ValueError:pass\n else:raise AssertionError("accepted preloaded framework")\n'
        f'assert sys.modules[{module!r}] is sentinel;'
        'assert "numpy" not in sys.modules and "mujoco" not in sys.modules')
    result = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr


def test_pure_fixtures_tolerate_collection_frameworks_without_unloading_them():
    code = ('import sys,types,pytest;'
        'sentinels={name:types.ModuleType(name) for name in ("torch","stable_baselines3","cv2")};'
        'sys.modules.update(sentinels);'
        'code=pytest.main(["--noconftest","-p","no:cacheprovider","-q",'
        '"tests/test_b22_numpy_backend38.py::test_new_no_framework_counter_covers_single32_probe_plus1600_batch1_exactly",'
        '"tests/test_b22_numpy_worker38.py::test_actual_cold_bootstrap_gate_requires_zero_counts_and_real_binding_proof[None]"]);'
        'assert code==0;assert all(sys.modules[name] is value for name,value in sentinels.items())')
    result = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=False,
        env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode == 0,result.stdout+result.stderr
