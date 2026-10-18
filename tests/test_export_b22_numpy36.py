"""Synthetic matrices/NPY/counters/lifecycle; no trained policy or Torch import."""
from __future__ import annotations

import importlib.util
import json
import math
import struct
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_numpy36', ROOT/'scripts/export_b22_numpy36.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def weights():
    return {name: np.zeros(shape, np.float32) for name, shape in m.WEIGHTS.items()}


def test_fixed_recipe_one_load_and_exact_total_real_nn_budget():
    plan = m.plan()
    assert plan['fixture_identity'] == {'bytes': 76032, 'sha256': 'b1468a1d91007cecb3d8c8a706fb1f44fb2857819eb96f57fd9c72c5cfeda834'}
    assert list(m.TICKS) == list(range(0, 1600, 50)) and len(m.HISTORY) == 4
    assert len(m.boundary_rows()) == 32 and len(m.TICKS)*len(m.HISTORY)+64 == 192
    assert m.LIMITS['torch_predict_calls'] == 193 and m.LIMITS['torch_predict_rows'] == 224
    assert m.LIMITS['numpy_predict_calls'] == m.LIMITS['numpy_predict_rows'] == 192
    assert m.LIMITS['ppo_load_calls'] == 1 and m.LIMITS['torch_load_calls'] == 3
    assert m.LIMITS['extra_benchmark_repeats'] == m.LIMITS['training_calls'] == m.LIMITS['physics_steps'] == 0
    assert m.LIMITS['start_effective_bytes'] == 6 << 30 and m.LIMITS['reserve_effective_bytes'] == 4 << 30
    assert m.LIMITS['worker_wall_s'] == 60 and m.LIMITS['total_wall_s'] == 180
    assert sum(math.prod(shape) for shape in m.WEIGHTS.values()) == 31376
    assert plan['qualification_granted'] is False and plan['execution_started'] is False


def test_default_cold_plan_never_imports_numeric_frameworks_or_creates_output(tmp_path):
    output = tmp_path/'not-created'
    code = ('import importlib.util,sys;s=importlib.util.spec_from_file_location("cold36","scripts/export_b22_numpy36.py");'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'assert not any(x in sys.modules for x in ("numpy","torch","mujoco","stable_baselines3","gymnasium"));'
            f'm.main(["--output",{str(output)!r}])')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'PLAN_ONLY' and not output.exists()


@pytest.mark.parametrize('module_name', ['torch', 'torch.nn', 'stable_baselines3', 'mujoco', 'gymnasium'])
def test_numpy_guard_blocks_indirect_import_before_module_execution(module_name):
    code = ('import importlib.util,sys;s=importlib.util.spec_from_file_location("guard36","scripts/export_b22_numpy36.py");'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'm.block_imports({"torch","stable_baselines3","mujoco","gymnasium"});'
            f'\ntry:__import__({module_name!r})\nexcept ImportError as e:assert "C36 blocked" in str(e)\nelse:raise AssertionError("guard failed")\n'
            'assert not any(x in sys.modules for x in ("torch","stable_baselines3","mujoco","gymnasium"))')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', [None, 'dtype', 'shape', 'fortran', 'truncate', 'version', 'code'])
def test_parent_npy_reader_numeric_only_fixed_shape_and_exact_length(change):
    array = np.arange(1601*99, dtype=np.float32).reshape(1601, 99)
    header = {'descr': '<f4', 'fortran_order': False, 'shape': (1601, 99)}
    if change == 'dtype':header['descr'] = '|O'
    elif change == 'shape':header['shape'] = (1600, 99)
    elif change == 'fortran':header['fortran_order'] = True
    text = repr(header).encode()+b'\n'
    if change == 'code':text = b'__import__("torch")\n'
    payload = b'\x93NUMPY\x01\x00'+struct.pack('<H', len(text))+text+array.tobytes()
    if change == 'truncate':payload = payload[:-1]
    elif change == 'version':payload = b'not-npy!'+payload[8:]
    if change is None:assert m.npy_observations(payload) == array.tobytes()
    else:
        with pytest.raises(ValueError):m.npy_observations(payload)


def test_parent_builder_selects_fixed_ticks_and_catches_changed_history(tmp_path, monkeypatch):
    probe = np.arange(32*99, dtype=np.float32).reshape(32, 99)/np.float32(1000.)
    (tmp_path/'final_probe_observations_f32.bin').write_bytes(probe.tobytes())
    config = {'shared_session': {'checkpoint_folder': str(tmp_path), 'checkpoint_manifest': {'files': {
        'final_probe_observations_f32.bin': m.identity(tmp_path/'final_probe_observations_f32.bin')}}}}
    sources, history = {}, []
    for i in range(4):
        values = np.full((1601, 99), i, np.float32)+np.arange(1601, dtype=np.float32)[:, None]/np.float32(1000.)
        path = tmp_path/f'states{i}.npz'
        np.savez(path, observation=values)
        sources[path.name] = m.identity(path)
        history.append(values)
    monkeypatch.setattr(m, 'HISTORY', sources)
    payload, origins = m.fixed_inputs(config, tmp_path)
    actual = np.frombuffer(payload, dtype='<f4').reshape(192, 99)
    assert actual[:32].tobytes() == probe.tobytes()
    for i, values in enumerate(history):assert actual[32+32*i:64+32*i].tobytes() == values[list(m.TICKS)].tobytes()
    assert [row['tick'] for row in origins[32:64]] == list(m.TICKS)
    assert actual[160:].tobytes() == np.asarray(m.boundary_rows(), np.float32).tobytes() and len(origins) == 192
    (tmp_path/'states0.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='historical'):m.fixed_inputs(config, tmp_path)


@pytest.mark.parametrize('change', [None, 'missing', 'extra', 'dtype', 'shape', 'nan'])
def test_only_six_finite_float32_actor_tensors_are_allowed(change):
    value = weights()
    if change == 'missing':value.pop('w1')
    elif change == 'extra':value['critic'] = np.zeros(1, np.float32)
    elif change == 'dtype':value['b2'] = value['b2'].astype(np.float64)
    elif change == 'shape':value['w1'] = value['w1'].T
    elif change == 'nan':value['w3'][0, 0] = np.nan
    if change is None:m.validate_weights(value, np)
    else:
        with pytest.raises(ValueError):m.validate_weights(value, np)


def test_synthetic_math_two_tanh_layers_and_clip_before_half_gain():
    value = weights()
    value['w1'][0, 0] = .3; value['b1'][0] = .1
    value['w2'][0, 0] = .7; value['b2'][0] = -.2
    value['w3'][0, 0] = .4; value['b3'][1:3] = [2., -2.]
    obs = np.zeros(99, np.float32);obs[0] = .5
    raw, gain = m.numpy_action(obs, value, np)
    expected = .4*math.tanh(.7*math.tanh(.3*.5+.1)-.2)
    assert abs(float(raw[0])-expected) < 1e-7
    assert raw[1] == 1. and raw[2] == -1. and gain[1] == .5 and gain[2] == -.5
    assert np.array_equal(gain, raw*np.float32(.5)) and raw.dtype == gain.dtype == np.float32 and obs[0] == .5


@pytest.mark.parametrize('change', ['shape', 'dtype', 'nan', 'outside'])
def test_synthetic_numpy_math_rejects_invalid_observation99(change):
    obs = np.zeros(99, np.float32)
    if change == 'shape':obs = obs.reshape(1, 99)
    elif change == 'dtype':obs = obs.astype(np.float64)
    elif change == 'nan':obs[0] = np.nan
    else:obs[0] = 5.01
    with pytest.raises(ValueError):m.numpy_action(obs, weights(), np)


@pytest.mark.parametrize('change', [None, 'raw_fails_half_passes', 'wrong_gain', 'nan', 'shape', 'keys'])
def test_original_clipped_tolerance_cannot_be_weakened_by_half_gain(change):
    reference = {name: np.zeros((192, 16), np.float32) for name in ('clipped', 'gain0p5')}
    actual = {name: value.copy() for name, value in reference.items()}
    if change == 'raw_fails_half_passes':
        actual['clipped'][0, 0] = 1.25e-6
        actual['gain0p5'][0, 0] = actual['clipped'][0, 0]*np.float32(.5)
    elif change == 'wrong_gain':actual['gain0p5'][0, 0] = 1e-7
    elif change == 'nan':actual['clipped'][0, 0] = np.nan
    elif change == 'shape':actual['clipped'] = actual['clipped'][:191]
    elif change == 'keys':actual.pop('gain0p5')
    if change is None:assert m.numerical_parity(reference, actual, np) == {'clipped': 0., 'gain0p5': 0.}
    else:
        with pytest.raises(ValueError):m.numerical_parity(reference, actual, np)


@pytest.fixture
def extractor(monkeypatch):
    for package in ('stable_baselines3', 'stable_baselines3.common'):
        fake = types.ModuleType(package)
        fake.__path__ = []
        monkeypatch.setitem(sys.modules, package, fake)
    class Linear:
        in_features = 128
        out_features = 16
    class Tanh:
        pass
    class Sequential:
        def __init__(self):self.layers = [Linear(), Tanh(), Linear(), Tanh()]
        def children(self):return iter(self.layers)
    class Flatten:
        start_dim = 1
        end_dim = -1
    class FlattenExtractor:
        features_dim = 99
        def parameters(self):return iter(())
        def children(self):return iter((Flatten(),))
    class ActorCriticPolicy:
        pass
    class DiagGaussianDistribution:
        pass
    for path, objects in {
        'stable_baselines3.common.distributions': {'DiagGaussianDistribution': DiagGaussianDistribution},
        'stable_baselines3.common.policies': {'ActorCriticPolicy': ActorCriticPolicy},
        'stable_baselines3.common.torch_layers': {'FlattenExtractor': FlattenExtractor},
    }.items():
        value = types.ModuleType(path)
        value.__dict__.update(objects)
        monkeypatch.setitem(sys.modules, path, value)
    class Tensor:
        def __init__(self, data):self.data, self.dtype, self.device = data, np.float32, SimpleNamespace(type='cpu')
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.data
    names = {'w1':'mlp_extractor.policy_net.0.weight','b1':'mlp_extractor.policy_net.0.bias',
             'w2':'mlp_extractor.policy_net.2.weight','b2':'mlp_extractor.policy_net.2.bias',
             'w3':'action_net.weight','b3':'action_net.bias'}
    state = {names[k]: Tensor(v) for k,v in weights().items()}
    policy = ActorCriticPolicy()
    feature = FlattenExtractor()
    policy.action_dist = DiagGaussianDistribution()
    policy.features_extractor = policy.pi_features_extractor = policy.vf_features_extractor = feature
    policy.share_features_extractor, policy.squash_output, policy.use_sde = True, False, False
    policy.action_net, policy.mlp_extractor = Linear(), SimpleNamespace(policy_net=Sequential())
    policy.state_dict = lambda: state
    model = SimpleNamespace(policy=policy, get_env=lambda:None, get_vec_normalize_env=lambda:None, device='cpu', observation_space='obs', action_space='action')
    fake_torch = SimpleNamespace(float32=np.float32, nn=SimpleNamespace(Linear=Linear, Tanh=Tanh, Sequential=Sequential, Flatten=Flatten))
    expected = [{'type':'Linear','in_features':99,'out_features':128},{'type':'Tanh'}, {'type':'Linear','in_features':128,'out_features':128},{'type':'Tanh'}]
    api = SimpleNamespace(_describe_box=lambda space,size,bound,label:{'dtype':'float32','shape':[size],'low':-bound,'high':bound},
                          _describe_linear_stack=lambda stack:expected, hash_state=lambda *a:m.POLICY_STATE_SHA)
    return model, fake_torch, api, state


@pytest.mark.parametrize('change', [None, 'feature', 'distribution', 'flatten', 'normalizer', 'squash', 'sde', 'extra', 'missing', 'dtype', 'hash', 'graph','stack_type','layer_type'])
def test_real_extractor_control_flow_exact_preprocess_graph_with_fake_types(extractor, change):
    model, torch, api, state = extractor
    if change == 'feature':model.policy.features_extractor = SimpleNamespace()
    elif change == 'distribution':model.policy.action_dist = SimpleNamespace()
    elif change == 'flatten':model.policy.features_extractor.features_dim = 98
    elif change == 'normalizer':model.get_vec_normalize_env = lambda:object()
    elif change == 'squash':model.policy.squash_output = True
    elif change == 'sde':model.policy.use_sde = True
    elif change == 'extra':state['action_net.extra'] = state['action_net.bias']
    elif change == 'missing':state.pop('action_net.bias')
    elif change == 'dtype':state['action_net.bias'].dtype = np.float64
    elif change == 'hash':api.hash_state = lambda *a:'0'*64
    elif change == 'graph':api._describe_linear_stack = lambda *a:[]
    elif change == 'stack_type':model.policy.mlp_extractor.policy_net = SimpleNamespace(children=lambda:iter(()))
    elif change == 'layer_type':model.policy.mlp_extractor.policy_net.layers[1] = type('Tanh',(),{})()
    if change is None:assert set(m.extract_actor(model, np, torch, api)) == set(m.WEIGHTS)
    else:
        with pytest.raises(ValueError):m.extract_actor(model, np, torch, api)


def fake_worker_record(output, release_path, stage):
    """Synthetic arrays/metadata only, for the independent stdlib saved gate."""
    output.mkdir(exist_ok=True)
    for filename, data in (('actor.npz', weights()), (stage+'_outputs.npz', {'clipped':np.zeros((192,16),np.float32),'gain0p5':np.zeros((192,16),np.float32)})):
        if not (output/filename).exists():np.savez(output/filename, **data)
    if not (output/'actor_manifest.json').exists():m.write_json(output/'actor_manifest.json', {'synthetic':'not-an-export'})
    proof = {'probe_rows':32,'probe_actions_byte_exact':True,'loaded_policy_state_sha256':m.POLICY_STATE_SHA,'metadata_sha256':m.METADATA_SHA}
    ledger = {'counts':{k:{'attempted':v,'returned':v} for k,v in m.API_LIMITS.items()}, 'critic_rows':0}
    ledger['counts']['predict']['rows_returned'] = 224
    return {'schema':'b22-numpy36-worker-v1','status':'COMPLETE','error':None,'stage':stage,'release_identity':m.identity(release_path),
            'physics_steps':0,'environment_constructions':0,'qualification_granted':False,
            'actual_nn_counts':{'ppo_load':1,'torch_load':3,'predict_calls':193,'predict_rows':224} if stage=='torch' else {'ppo_load':0,'torch_load':0,'predict_calls':192,'predict_rows':192},
            'actor_identity':m.identity(output/'actor.npz'),'actor_manifest_identity':m.identity(output/'actor_manifest.json'),
            'outputs_identity':m.identity(output/(stage+'_outputs.npz')),'strict_loader':proof,'api_ledger':ledger,
            'common192_predict_wall_s':.1,'cold_to_workload_end_s':1.,'peak_rss_kib':10000,
            'forbidden_modules_absent':True,'parity_max_absolute_error':{'clipped':0.,'gain0p5':0.}}


@pytest.mark.parametrize('stage', ['torch','numpy'])
@pytest.mark.parametrize('change', [None,'count','row','identity','physics','time','rss','artifact','api','probe','error'])
def test_saved_worker_gate_rejects_identity_counts_missing_measurements_and_physics(tmp_path, stage, change):
    path = tmp_path/'release.json';m.write_json(path, {'synthetic':'release'})
    release = {'plan':{'output':str(tmp_path)}}
    fake_worker_record(tmp_path, path, 'torch')
    record = fake_worker_record(tmp_path, path, stage)
    if change == 'count':record['actual_nn_counts']['predict_calls'] -= 1
    elif change == 'row':record['actual_nn_counts']['predict_rows'] += 1
    elif change == 'identity':record['release_identity'] = {}
    elif change == 'physics':record['physics_steps'] = 1
    elif change == 'time':record['common192_predict_wall_s'] = 2.
    elif change == 'rss':record['peak_rss_kib'] = 0
    elif change == 'artifact':(tmp_path/'actor.npz').write_bytes(b'changed')
    elif change == 'api':record['api_ledger']['counts']['torch_load']['returned'] = 2
    elif change == 'probe':record['strict_loader']['loaded_policy_state_sha256'] = '0'*64
    elif change == 'error':record['status'] = 'INCOMPLETE'
    invalid = change is not None and not (stage=='numpy' and change in ('api','probe'))
    if invalid:
        with pytest.raises(ValueError):m.verify_worker(release, path, stage, record)
    else:m.verify_worker(release, path, stage, record)


def test_parent_recomputes_numerical_error_despite_declared_zero_and_updated_sha(tmp_path):
    path = tmp_path/'release.json';m.write_json(path, {'synthetic':'release'})
    fake_worker_record(tmp_path, path, 'torch')
    record = fake_worker_record(tmp_path, path, 'numpy')
    raw = np.zeros((192,16), np.float32);raw[3,4] = 1.25e-6
    np.savez(tmp_path/'numpy_outputs.npz', clipped=raw, gain0p5=raw*np.float32(.5))
    record['outputs_identity'] = m.identity(tmp_path/'numpy_outputs.npz')
    with pytest.raises(ValueError, match='saved numerical'):m.verify_worker({'plan':{'output':str(tmp_path)}}, path, 'numpy', record)


@pytest.mark.parametrize('change', [None,'parent','nonce','argv','release','previous'])
def test_worker_actual_parent_nonce_exact_argv_and_predecessor(tmp_path, monkeypatch, change):
    path = tmp_path/'release.json';m.write_json(path, {'synthetic':'release'})
    release = {'plan':{'output':str(tmp_path)}}
    monkeypatch.setattr(m, 'check_release', lambda p:release)
    monkeypatch.setenv('B22_NUMPY36_NONCE','new36')
    journal = {'status':'RUNNING','active_stage':'numpy','supervisor_pid':m.os.getppid(),'release_identity':m.identity(path),
               'stages':{'torch':{'status':'COMPLETE'},'numpy':{'status':'STARTED','nonce':'new36','command':m.worker_argv(path,'numpy')}}}
    if change == 'parent':journal['supervisor_pid'] += 1
    elif change == 'nonce':journal['stages']['numpy']['nonce'] = 'stale'
    elif change == 'argv':journal['stages']['numpy']['command'].append('--benchmark-again')
    elif change == 'release':journal['release_identity'] = {}
    elif change == 'previous':journal['stages']['torch']['status'] = 'INCOMPLETE'
    m.write_json(tmp_path/'report.json', journal)
    if change is None:assert m.validate_worker_launch(path, 'numpy') == release
    else:
        with pytest.raises(ValueError):m.validate_worker_launch(path,'numpy')


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    """Run the real parent with synthetic artifacts, no process/NN/framework load."""
    root = tmp_path/'checkout'
    (root/'results').mkdir(parents=True)
    for name in (*m.AUTHORED, m.MANIFEST, 'host.py', 'scripts/run_b22_suite31.py'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic source only\n')
    config = {'host':str(root/'host.py'), 'source_hashes':{str(root/'host.py'):m.identity(root/'host.py')},
              'runtime_environment':{'PYTHONPATH':'synthetic-not-used'}}
    (root/m.MANIFEST).write_text(json.dumps(config))
    monkeypatch.setattr(m, 'ROOT', root)
    monkeypatch.setattr(m, 'REGISTERED_ROOT', root)
    monkeypatch.setattr(m, 'MANIFEST_SHA', m.identity(root/m.MANIFEST)['sha256'])
    monkeypatch.setattr(m, 'PINS', {'scripts/run_b22_suite31.py':m.identity(root/'scripts/run_b22_suite31.py')['sha256']})
    monkeypatch.setattr(m, 'HISTORY', {})
    payload = bytes(192*99*4)
    monkeypatch.setattr(m, 'FIXTURE', {'bytes':len(payload),'sha256':m.hashlib.sha256(payload).hexdigest()})
    monkeypatch.setattr(m, 'fixed_inputs', lambda config:(payload, [{'synthetic':i} for i in range(192)]))
    source = {'git_commit':'synthetic', 'authored':{str(root/name):m.identity(root/name) for name in m.AUTHORED}}
    monkeypatch.setattr(m, 'source_identity', lambda:source)
    monkeypatch.setattr(m.resource, 'setrlimit', lambda *a:None)
    monkeypatch.setattr(m.os, 'getpriority', lambda *a:10)
    original_signal = m.signal
    monkeypatch.setattr(m, 'signal', SimpleNamespace(**{name:getattr(original_signal,name) for name in ('SIGINT','SIGTERM','SIGXCPU','SIGALRM','SIG_IGN','ITIMER_REAL')},
                                                   signal=lambda *a:None,setitimer=lambda *a:None))
    clock = [0.]
    monkeypatch.setattr(m, 'time', SimpleNamespace(monotonic=lambda:clock[0]))
    state = SimpleNamespace(root=root, output=root/'results/b22_numpy36_fake', calls=[], cleanups=[], mode=None, subreaper=False, original_signal=original_signal)
    def memory():
        if state.mode == 'startup':return 5 << 30
        if state.mode == 'numpy_admission' and state.calls:return 5 << 30
        return 8 << 30
    def subreaper():state.subreaper = True
    def cleanup(pid, process):
        state.cleanups.append((pid,process))
        if state.mode == 'late_cleanup':clock[0] = 181.
        return {'remaining':{'99999':{'synthetic':True}} if state.mode == 'cleanup' else {}}
    host = SimpleNamespace(subreaper=subreaper, clean_descendants=cleanup)
    def supervise(command, log, *, host, sample, **kwargs):
        assert state.subreaper and not state.cleanups
        assert 'factory' in kwargs and kwargs['limit'] == 60 and kwargs['deadline'] == 170.
        stage = command[command.index('--worker')+1]
        state.calls.append(stage)
        if state.mode == 'signal':raise InterruptedError('synthetic SIGTERM before child handle')
        if state.mode == 'cpu_signal':
            assert callable(original_signal.getsignal(original_signal.SIGXCPU)), 'SIGXCPU must be handled before real signal'
            original_signal.raise_signal(original_signal.SIGXCPU)
        if state.mode == 'ram':
            shared.memory = lambda:3 << 30
            sample(False)
        record = fake_worker_record(state.output, state.output/'release.json', stage)
        if state.mode == 'partial':
            record.update(status='INCOMPLETE',actual_nn_counts=None,known_actual_nn_counts={'predict':{'attempted':14,'returned':13}})
        m.write_json(state.output/(stage+'_report.json'), record)
        if stage == 'numpy' and state.mode == 'source':(root/m.AUTHORED[0]).write_text('changed source\n')
        return 1 if state.mode == 'partial' else 0
    shared = SimpleNamespace(memory=memory, supervise=supervise)
    def load_pinned(name, path, expected):
        assert m.identity(path) == expected
        return host if name == 'numpy36_cleanup_host' else shared
    monkeypatch.setattr(m, 'load_pinned', load_pinned)
    return state


@pytest.mark.parametrize('mode', [None,'signal','ram','partial','numpy_admission','cleanup','late_cleanup','source'])
def test_real_parent_serial_completion_and_bounded_failure_receipts(supervisor, mode):
    supervisor.mode = mode
    report = m.execute(supervisor.output)
    assert report == m.read_json(supervisor.output/'report.json')
    assert supervisor.cleanups == [(m.os.getpid(),None)] and report['qualification_granted'] is False
    if mode is None:
        assert supervisor.calls == ['torch','numpy'] and report['status'] == 'COMPLETE'
        assert report['actual_nn_counts'] == {'ppo_load':1,'torch_load':3,'torch_predict_calls':193,'torch_predict_rows':224,
                                             'numpy_predict_calls':192,'numpy_predict_rows':192}
        assert report['parity_max_absolute_error'] == {'clipped':0.,'gain0p5':0.}
    else:
        assert report['status'] == 'INCOMPLETE' and report['actual_nn_counts'] is None and report['error']
        assert supervisor.calls == (['torch','numpy'] if mode in ('cleanup','late_cleanup','source') else ['torch'])
        if mode == 'partial':assert report['stages']['torch']['known_nn_prefix'] == {'predict':{'attempted':14,'returned':13}}
        elif mode in ('signal','ram'):assert report['stages']['torch']['known_nn_prefix'] is None


def test_startup_resource_not_ready_spawns_nothing_and_does_not_create_output(supervisor):
    supervisor.mode = 'startup'
    with pytest.raises(ValueError, match='NOT_READY'):m.execute(supervisor.output)
    assert not supervisor.output.exists() and supervisor.calls == supervisor.cleanups == []


def test_existing_output_is_never_reused_or_retried(supervisor):
    supervisor.output.mkdir()
    with pytest.raises(ValueError, match='no retry'):m.execute(supervisor.output)
    assert supervisor.calls == supervisor.cleanups == []


def test_parent_real_cpu_soft_limit_signal_records_unknown_counts_and_cleans_owned(supervisor, monkeypatch):
    monkeypatch.setattr(m, 'signal', supervisor.original_signal)
    supervisor.mode = 'cpu_signal'
    previous = m.signal.getsignal(m.signal.SIGXCPU)
    report = m.execute(supervisor.output)
    assert report['status'] == 'INCOMPLETE' and 'SIGXCPU' in report['error'] and report['actual_nn_counts'] is None
    assert supervisor.calls == ['torch'] and supervisor.cleanups == [(m.os.getpid(),None)]
    assert report['stages']['torch']['known_nn_prefix'] is None
    assert m.signal.getsignal(m.signal.SIGXCPU) == previous
