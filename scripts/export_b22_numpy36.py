"""Plan or supervise one fixed B22 actor-only numerical/cold-start comparison.

The parent is stdlib-only. Exactly one supervised Torch child performs the
original strict load; the other child forbids Torch/SB3 and runs only NumPy.
Neither path imports a simulator or constructs an environment.
"""
from __future__ import annotations

import time

_COLD_STARTED = time.perf_counter()

import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
import platform
import resource
import signal
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_ROOT = Path('/home/lyh/wheel-legged-control-lab')
AUTHORED = ('scripts/export_b22_numpy36.py', 'tests/test_export_b22_numpy36.py', 'docs/b22_numpy36.md')
MANIFEST = 'runtime/d1_b22/runtime_manifest_v27.json'
MANIFEST_SHA = 'd3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09'
MODEL_SHA = '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'
METADATA_SHA = 'af700d1f2612e441c8f0560cca38c77b6a854526d3e841e544ef9d5321672830'
POLICY_STATE_SHA = '59440de0c5beb102cec0d292838fb9b125eca7eb56bbffc06d1eaec46ec7aa48'
PINS = {
    'scripts/run_b22_suite31.py': 'f783b4d7e8380c94bd7a4755e73434179cb5c3f6a95bbcca336e235a43320c2e',
    'scripts/b22_suite31_contract.py': '21386cfb3b73f0d947f825f00dd367c7443f3ca29738dfec5768925f30070203',
    'runtime/d1_b22/research/rl11/rl16_learning_11.py': '837b1b7c814fcb174cc8f1d8ce0ad1bfcbf623e2cc4aba63ad36c3304f5ff673',
    'runtime/d1_b22/research/rl11/budget_spec_11.py': '606e9eaf6b979f9616a5721b044067670c6a34946cd2ea84ce058e8722dcc2d6',
    'runtime/d1_b22/research/course_impl08/residual16_math_08.py': 'ff66faae8cd29779f6876a8abac2d9e0c92e0913484097059b76b5d43dd8d635',
    'runtime/d1_b22/research/continuation22/runtime_support22.py': 'ffed9b03e0baa329f87691fe3376cf93d9db916996df075c2da5193cde677411',
}
HISTORY = {
    'results/b22_suite31_20261018_01/flat_0p6_B/episode_0/states.npz': {'bytes': 1711364, 'sha256': 'edf784609e806c01b17174e0fb6dc475c2d4d9c172908e035161e54e67ccc20c'},
    'results/b22_suite31_20261018_01/yaw_1p2_B/episode_0/states.npz': {'bytes': 1711364, 'sha256': '1cdf321cbd6ee3e2ed9d33fc966a9f9e112e9e2674d3f3481dfeae7acef17912'},
    'results/b22_suite31_20261018_01/rough_0p35_B/episode_0/states.npz': {'bytes': 1711364, 'sha256': '1581666ab61f93a35fd85b6296ab7fb1aa2a49a435e1c64484e4bc052ad615f2'},
    'results/b22_inference32_20261018_01/gated/episode_0/states.npz': {'bytes': 1711364, 'sha256': '6fc05b51581b19ffcdb8d0153cc56b68f65d89ef54610c6fddeef49f43d067ae'},
}
TICKS = tuple(range(0, 1600, 50))
FIXTURE = {'bytes': 76032, 'sha256': 'b1468a1d91007cecb3d8c8a706fb1f44fb2857819eb96f57fd9c72c5cfeda834'}
WEIGHTS = {'w1': (128, 99), 'b1': (128,), 'w2': (128, 128), 'b2': (128,), 'w3': (16, 128), 'b3': (16,)}
API_LIMITS = {'load': 1, 'torch_load': 3, 'predict': 193, 'save': 0, 'learn': 0, 'train': 0,
              'forward': 0, 'evaluate_actions': 0, 'predict_values': 0, 'backward': 0}
LIMITS = {'start_effective_bytes': 6 << 30, 'reserve_effective_bytes': 4 << 30,
          'total_wall_s': 180, 'worker_wall_s': 60, 'worker_cpu_soft_s': 30, 'worker_cpu_hard_s': 35,
          'parent_cpu_soft_s': 60, 'parent_cpu_hard_s': 65,
          'required_minimum_nice': 10, 'torch_predict_calls': 193, 'torch_predict_rows': 224,
          'numpy_predict_calls': 192, 'numpy_predict_rows': 192, 'ppo_load_calls': 1, 'torch_load_calls': 3,
          'physics_steps': 0, 'environment_constructions': 0, 'training_calls': 0, 'extra_benchmark_repeats': 0}
ABSOLUTE_LIMIT = 1e-6
SIMULATORS = {'mujoco', 'engine_binding', 'dm_control', 'pybullet'}


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: '+value)
    result = json.loads(Path(path).read_text(), parse_constant=invalid)
    require(isinstance(result, dict), 'JSON object required')
    return result


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def load_pinned(name, path, expected):
    require(identity(path) == expected, 'source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def plan(output=None):
    return {'schema': 'b22-numpy36-plan-v1', 'status': 'PLAN_ONLY',
            'owner_instruction': '把这些优化点，在这台电脑能力之内的都完成',
            'output': None if output is None else str(Path(output).resolve()), 'execution_started': False,
            'limits': LIMITS, 'absolute_tolerance': ABSOLUTE_LIMIT, 'gain_after_clip': .5,
            'fixture_recipe': {'original_probe_rows': 32, 'history': HISTORY, 'ticks_per_history': list(TICKS),
                               'boundary_rows': 32, 'total_rows': 192, 'shape': [192, 99], 'dtype': 'little-endian float32'},
            'fixture_identity': FIXTURE,
            'stages': ['torch', 'numpy'], 'retry_permitted': False, 'qualification_granted': False,
            'scope': 'one strict PPO load/probe and common192 batch1 work; one independent cold NumPy process; no physics/control claim'}


def boundary_rows():
    rows = [[0.]*99, [5.]*99, [-5.]*99,
            [5. if i % 2 == 0 else -5. for i in range(99)],
            [-5. if i % 2 == 0 else 5. for i in range(99)],
            [-5.+10.*i/98. for i in range(99)], [5.-10.*i/98. for i in range(99)],
            [float.fromhex('0x1.0p-126')*(1. if i % 2 else -1.) for i in range(99)]]
    for k in range(24):
        row = [0.]*99
        row[(4*k) % 99] = 5. if k % 2 == 0 else -5.
        rows.append(row)
    return rows


def npy_observations(payload):
    """Read only this frozen numeric NPY schema; parent never imports NumPy."""
    require(payload[:8] == b'\x93NUMPY\x01\x00', 'only NPY1 observation99 is supported')
    length = struct.unpack('<H', payload[8:10])[0]
    require(length <= 4096, 'oversize NPY header')
    header = ast.literal_eval(payload[10:10+length].decode('latin1'))
    require(header == {'descr': '<f4', 'fortran_order': False, 'shape': (1601, 99)}, 'saved observation NPY schema differs')
    data = payload[10+length:]
    require(len(data) == 1601*99*4, 'saved observation NPY length differs')
    return data


def saved_output_rows(path):
    """Recompute artifact/output checks in the stdlib parent, without NN imports."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {'clipped.npy', 'gain0p5.npy'}, 'C36 output members differ')
        for name in ('clipped', 'gain0p5'):
            data = archive.read(name+'.npy')
            require(data[:8] == b'\x93NUMPY\x01\x00', 'C36 output NPY version differs')
            length = struct.unpack('<H', data[8:10])[0]
            require(length <= 4096 and ast.literal_eval(data[10:10+length].decode('latin1'))
                    == {'descr': '<f4', 'fortran_order': False, 'shape': (192, 16)}
                    and len(data) == 10+length+192*16*4, 'C36 output NPY schema differs')
            values = struct.unpack('<'+str(192*16)+'f', data[10+length:])
            require(all(math.isfinite(v) and -1. <= v <= 1. for v in values), 'C36 output is nonfinite/outside action box')
            result[name] = values
    expected = struct.pack('<'+str(192*16)+'f', *(v*.5 for v in result['clipped']))
    actual = struct.pack('<'+str(192*16)+'f', *result['gain0p5'])
    require(expected == actual, 'C36 saved gain is not after clipped action')
    return result


def saved_parity(output):
    reference, actual = (saved_output_rows(Path(output)/(name+'_outputs.npz')) for name in ('torch', 'numpy'))
    result = {name: max(abs(a-b) for a,b in zip(reference[name], actual[name])) for name in reference}
    require(all(value <= ABSOLUTE_LIMIT for value in result.values()), 'C36 saved numerical error exceeds 1e-6')
    return result


def fixed_inputs(config, root=ROOT):
    folder = Path(config['shared_session']['checkpoint_folder'])
    manifest = config['shared_session']['checkpoint_manifest']
    probe = folder/'final_probe_observations_f32.bin'
    require(identity(probe) == manifest['files'][probe.name], 'original32 probe changed')
    chunks = [probe.read_bytes()]
    origins = [{'kind': 'original32', 'row': i} for i in range(32)]
    require(len(chunks[0]) == 32*99*4, 'original probe shape differs')
    for name, expected in HISTORY.items():
        path = root/name
        require(identity(path) == expected, 'fixed historical input changed: '+name)
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo('observation.npy')
            require(info.file_size <= 1601*99*4+4106, 'unexpected observation archive size')
            values = npy_observations(archive.read(info))
        for tick in TICKS:
            chunks.append(values[tick*99*4:(tick+1)*99*4])
            origins.append({'kind': 'history', 'path': name, 'tick': tick})
    for index, row in enumerate(boundary_rows()):
        chunks.append(struct.pack('<99f', *row))
        origins.append({'kind': 'boundary', 'row': index})
    payload = b''.join(chunks)
    values = struct.unpack('<'+str(192*99)+'f', payload)
    require(len(origins) == 192 and len(payload) == 192*99*4
            and all(math.isfinite(v) and -5. <= v <= 5. for v in values), 'invalid fixed192 observation cohort')
    return payload, origins


def source_identity():
    subprocess.run(['git', 'ls-files', '--error-unmatch', *AUTHORED], cwd=ROOT, check=True, stdout=subprocess.DEVNULL, timeout=5)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *AUTHORED], cwd=ROOT, check=True, timeout=5)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip(),
            'authored': {str(ROOT/name): identity(ROOT/name) for name in AUTHORED}}


def check_release(path, *, check_all_inputs=True):
    release = read_json(path)
    require(release['schema'] == 'b22-numpy36-release-v1' and release['plan'] == plan(release['plan']['output']), 'C36 fixed plan differs')
    output = Path(release['plan']['output'])
    require(output.parent == ROOT/'results' and output.name.startswith('b22_numpy36_') and Path(path) == output/'release.json', 'C36 fixed output differs')
    require(identity(ROOT/MANIFEST)['sha256'] == MANIFEST_SHA, 'v27 identity changed')
    config = read_json(ROOT/MANIFEST)
    expected = {**config['source_hashes'], **release['source']['authored'],
                **{str(ROOT/name): entry for name, entry in HISTORY.items()}, str(ROOT/MANIFEST): identity(ROOT/MANIFEST),
                **{str(ROOT/name): {'bytes': (ROOT/name).stat().st_size, 'sha256': digest} for name, digest in PINS.items()}}
    require(set(release['source']['authored']) == {str(ROOT/name) for name in AUTHORED}
            and release['inputs'] == expected, 'C36 exact source/input closure differs')
    if check_all_inputs:
        for name, bound in release['inputs'].items():
            require(not Path(name).is_symlink() and identity(name) == bound, 'C36 source/input changed: '+name)
    require(identity(output/'inputs.f32.bin') == release['fixture_identity'] == FIXTURE
            and identity(output/'input_origins.json') == release['origins_identity'], 'C36 fixed fixture changed')
    return release


class BlockImports(importlib.abc.MetaPathFinder):
    def __init__(self, names):
        self.names = set(names)
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in self.names:
            raise ImportError('C36 blocked dependency: '+fullname)


def block_imports(names):
    require(not any(name.partition('.')[0] in names for name in sys.modules), 'worker was not a cold dependency boundary')
    guard = BlockImports(names)
    sys.meta_path.insert(0, guard)
    return guard


def dependency_origins(inputs, prefixes):
    """Bind actually imported numeric modules, including two Torch aliases."""
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name.partition('.')[0] not in prefixes:
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        if name in ('torch.ops', 'torch.classes'):
            defining = 'torch._ops' if name == 'torch.ops' else 'torch._classes'
            definition = sys.modules.get(defining)
            require(definition is not None and module is getattr(definition, name.split('.')[1]), 'C36 synthetic Torch alias differs')
            filename = definition.__file__
        path = Path(filename).resolve()
        require(path.is_absolute() and str(path) in inputs and identity(path) == inputs[str(path)], 'C36 unbound numeric import: '+name)
        origins[name] = str(path)
    require(all(name in origins for name in prefixes), 'C36 numeric dependency root missing')
    return origins


def validate_weights(weights, np):
    require(set(weights) == set(WEIGHTS), 'exactly six actor tensors are required')
    for name, shape in WEIGHTS.items():
        row = weights[name]
        require(isinstance(row, np.ndarray) and row.shape == shape and row.dtype == np.float32
                and bool(np.isfinite(row).all()), 'invalid finite float32 actor tensor: '+name)


def numpy_action(observation, weights, np):
    require(observation.shape == (99,) and observation.dtype == np.float32
            and bool(np.isfinite(observation).all()) and bool((abs(observation) <= 5.).all()), 'invalid batch1 observation99')
    hidden = np.tanh(observation.reshape(1, 99)@weights['w1'].T+weights['b1'])
    hidden = np.tanh(hidden@weights['w2'].T+weights['b2'])
    clipped = np.clip(hidden@weights['w3'].T+weights['b3'], np.float32(-1.), np.float32(1.)).reshape(16)
    return clipped, clipped*np.float32(.5)


def extract_actor(model, np, torch, api):
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import FlattenExtractor
    policy = model.policy
    require(type(policy) is ActorCriticPolicy and type(policy.action_dist) is DiagGaussianDistribution
            and type(policy.features_extractor) is FlattenExtractor and policy.share_features_extractor is True
            and policy.pi_features_extractor is policy.vf_features_extractor is policy.features_extractor
            and len(list(policy.features_extractor.parameters())) == 0
            and policy.squash_output is False and policy.use_sde is False
            and model.get_env() is None and model.get_vec_normalize_env() is None and str(model.device) == 'cpu',
            'C36 requires exact flat/unsquashed/stateless mean actor without external normalization')
    flatten = list(policy.features_extractor.children())
    require(policy.features_extractor.features_dim == 99 and len(flatten) == 1 and type(flatten[0]) is torch.nn.Flatten
            and flatten[0].start_dim == 1 and flatten[0].end_dim == -1, 'C36 actual FlattenExtractor has another preprocessing graph')
    require(api._describe_box(model.observation_space, 99, 5., 'obs99') == {'dtype': 'float32', 'high': 5., 'low': -5., 'shape': [99]}
            and api._describe_box(model.action_space, 16, 1., 'actions16') == {'dtype': 'float32', 'high': 1., 'low': -1., 'shape': [16]}, 'C36 spaces differ')
    expected = [{'type': 'Linear', 'in_features': 99, 'out_features': 128}, {'type': 'Tanh'},
                {'type': 'Linear', 'in_features': 128, 'out_features': 128}, {'type': 'Tanh'}]
    stack = policy.mlp_extractor.policy_net
    require(type(stack) is torch.nn.Sequential
            and tuple(type(layer) for layer in stack.children()) == (torch.nn.Linear, torch.nn.Tanh, torch.nn.Linear, torch.nn.Tanh)
            and api._describe_linear_stack(stack) == expected
            and type(policy.action_net) is torch.nn.Linear
            and policy.action_net.in_features == 128 and policy.action_net.out_features == 16,
            'C36 exact actor graph differs')
    tensor_names = {'w1': 'mlp_extractor.policy_net.0.weight', 'b1': 'mlp_extractor.policy_net.0.bias',
                    'w2': 'mlp_extractor.policy_net.2.weight', 'b2': 'mlp_extractor.policy_net.2.bias',
                    'w3': 'action_net.weight', 'b3': 'action_net.bias'}
    state = policy.state_dict()
    selected = {name for name in state if name.startswith(('mlp_extractor.policy_net.', 'action_net.'))}
    require(selected == set(tensor_names.values()), 'C36 actor has extra or missing state tensors')
    weights = {}
    for name, tensor in tensor_names.items():
        require(state[tensor].dtype == torch.float32 and state[tensor].device.type == 'cpu', 'C36 actor tensor dtype/device differs')
        weights[name] = state[tensor].detach().cpu().numpy().copy()
    validate_weights(weights, np)
    require(api.hash_state('policy', state) == POLICY_STATE_SHA, 'C36 full policy identity differs')
    return weights


def output_action(value, np):
    clipped, state = value
    row = np.asarray(clipped)
    require(state is None and row.shape == (16,) and row.dtype == np.float32
            and bool(np.isfinite(row).all()) and bool((abs(row) <= 1.).all()), 'C36 reference output/state differs')
    return row.copy(), row*np.float32(.5)


def numerical_parity(reference, actual, np):
    errors = {}
    require(set(reference) == set(actual) == {'clipped', 'gain0p5'}, 'C36 two output variants required')
    for name in reference:
        a, b = reference[name], actual[name]
        require(a.shape == b.shape == (192, 16) and a.dtype == b.dtype == np.float32
                and bool(np.isfinite(a).all()) and bool(np.isfinite(b).all()), 'invalid C36 parity outputs')
        maximum = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
        require(maximum <= ABSOLUTE_LIMIT, 'C36 '+name+' absolute error exceeds 1e-6')
        errors[name] = maximum
    require(np.array_equal(actual['gain0p5'], actual['clipped']*np.float32(.5))
            and np.array_equal(reference['gain0p5'], reference['clipped']*np.float32(.5)), 'gain must follow clipped action')
    return errors


def worker_argv(release_path, stage):
    return ['/usr/bin/python3', '-B', str(ROOT/'scripts/export_b22_numpy36.py'), '--worker', stage, '--release', str(release_path)]


def validate_worker_launch(release_path, stage):
    release = check_release(release_path)
    output = Path(release['plan']['output'])
    journal = read_json(output/'report.json')
    record = journal['stages'].get(stage, {})
    require(journal['status'] == 'RUNNING' and journal['active_stage'] == stage
            and journal['supervisor_pid'] == os.getppid() and journal['release_identity'] == identity(release_path)
            and record.get('status') == 'STARTED' and record.get('nonce') == os.environ.get('B22_NUMPY36_NONCE')
            and bool(record.get('nonce')) and record.get('command') == worker_argv(release_path, stage), 'C36 live parent/nonce required')
    if stage == 'numpy':require(journal['stages'].get('torch', {}).get('status') == 'COMPLETE', 'C36 reference stage incomplete')
    return release


def worker(release_path, stage):
    require(stage in ('torch', 'numpy'), 'C36 worker mode differs')
    release = validate_worker_launch(release_path, stage)
    output = Path(release['plan']['output'])
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS['worker_cpu_soft_s'], LIMITS['worker_cpu_hard_s']))
    forbidden = SIMULATORS | ({'torch', 'stable_baselines3', 'gymnasium', 'gym', 'jax', 'tensorflow'} if stage == 'numpy' else set())
    block_imports(forbidden)
    require(not any(name in sys.modules for name in ('torch', 'stable_baselines3')), 'C36 expected a fresh cold process')
    report = {'schema': 'b22-numpy36-worker-v1', 'status': 'INCOMPLETE', 'stage': stage, 'error': None,
              'release_identity': identity(release_path), 'actual_nn_counts': None, 'known_actual_nn_counts': {},
              'physics_steps': 0, 'environment_constructions': 0, 'qualification_granted': False}
    counter = values = None
    attempted = returned = 0
    def interrupted(signum, frame):
        raise InterruptedError('C36 worker interrupted: '+signal.Signals(signum).name)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGXCPU):signal.signal(sig, interrupted)
    try:
        import numpy as np
        inputs = np.frombuffer((output/'inputs.f32.bin').read_bytes(), dtype='<f4').reshape(192, 99).copy()
        require(bool(np.isfinite(inputs).all()) and bool((abs(inputs) <= 5.).all()), 'C36 finite box99 inputs required')
        original_input = inputs.tobytes()
        if stage == 'torch':
            for directory in ('rl11', 'course_impl08', 'continuation22'):
                sys.path.insert(0, str(ROOT/'runtime/d1_b22/research'/directory))
            import gymnasium
            import stable_baselines3 as sb3
            import torch
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:torch.set_num_interop_threads(1)
            import rl16_learning_11 as api
            from runtime_support22 import ModelCounter
            config = read_json(ROOT/MANIFEST)
            manifest = config['shared_session']['checkpoint_manifest']
            folder = config['shared_session']['checkpoint_folder']
            require(manifest['files']['final_model.zip']['sha256'] == MODEL_SHA and manifest['metadata_sha256'] == METADATA_SHA, 'C36 fixed B22 checkpoint differs')
            counter = ModelCounter(False, dict(API_LIMITS))
            with counter:
                counter.phase = 'probe'
                model, proof = api.load_and_verify_final(folder, manifest, return_model=True)
                require(proof['probe_rows'] == 32 and proof['probe_actions_byte_exact'] is True
                        and proof['loaded_policy_state_sha256'] == POLICY_STATE_SHA, 'C36 original strict probe failed')
                weights = extract_actor(model, np, torch, api)
                counter.phase = 'common192'
                started = time.perf_counter()
                clipped, gained = [], []
                for row in inputs:
                    a, b = output_action(model.predict(row, deterministic=True), np)
                    clipped.append(a); gained.append(b)
                common_end = time.perf_counter()
                values = {'clipped': np.asarray(clipped, dtype=np.float32), 'gain0p5': np.asarray(gained, dtype=np.float32)}
            require(api.hash_state('policy', model.policy.state_dict()) == POLICY_STATE_SHA, 'C36 inference changed policy state')
            counts = counter.summary()
            require(all(counts['counts'].get(k, {}).get('attempted', 0) == counts['counts'].get(k, {}).get('returned', 0) == v for k, v in API_LIMITS.items())
                    and counts['counts']['predict']['rows_returned'] == 224 and counts['critic_rows'] == 0, 'C36 exact Torch API/row counts differ')
            report.update(actual_nn_counts={'ppo_load': 1, 'torch_load': 3, 'predict_calls': 193, 'predict_rows': 224},
                          api_ledger=counts, strict_loader=proof,
                          versions={'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                                    'sb3': sb3.__version__, 'gymnasium': gymnasium.__version__})
            document = read_json(Path(folder)/'final_metadata.json')
            require(all(report['versions'][key] == document[key+'_version'] for key in ('numpy', 'torch', 'sb3')), 'C36 versions differ from fixed training')
            # Export I/O is deliberately after both common-workload and cold-to-workload timing endpoints.
            report.update(common192_predict_wall_s=common_end-started, cold_to_workload_end_s=common_end-_COLD_STARTED,
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            with (output/'actor.npz').open('xb') as stream:np.savez(stream, **weights)
            bundle = {'schema': 'b22-numpy36-actor-v1', 'weights': identity(output/'actor.npz'), 'tensor_shapes': {k:list(v) for k,v in WEIGHTS.items()},
                      'dtype': 'float32', 'network': [99, 128, 128, 16], 'activation': 'Tanh', 'output': 'clip_minus1_plus1_then_optional_half_gain',
                      'input_contract': 'already_prepared_saved_observation99; no additional normalization in current predict path; not a raw sensor builder',
                      'model_sha256': MODEL_SHA, 'metadata_sha256': METADATA_SHA, 'policy_state_sha256': POLICY_STATE_SHA,
                      'original_probe_files': manifest['files'], 'strict_loader': proof, 'versions': report['versions'],
                      'release_identity': identity(release_path), 'fixture_identity': release['fixture_identity'],
                      'source': release['source'], 'qualification_granted': False}
            write_json(output/'actor_manifest.json', bundle)
            report.update(actor_manifest_identity=identity(output/'actor_manifest.json'), actor_identity=identity(output/'actor.npz'))
        else:
            journal = read_json(output/'report.json')
            require(journal['stages']['torch']['report_identity'] == identity(output/'torch_report.json'), 'C36 prior Torch report changed')
            torch_report = read_json(output/'torch_report.json')
            require(torch_report['actor_manifest_identity'] == identity(output/'actor_manifest.json')
                    and torch_report['actor_identity'] == identity(output/'actor.npz')
                    and torch_report['outputs_identity'] == identity(output/'torch_outputs.npz'), 'C36 reference artifacts changed')
            bundle = read_json(output/'actor_manifest.json')
            require(bundle['schema'] == 'b22-numpy36-actor-v1' and bundle['release_identity'] == identity(release_path)
                    and bundle['weights'] == identity(output/'actor.npz') and bundle['model_sha256'] == MODEL_SHA
                    and bundle['metadata_sha256'] == METADATA_SHA and bundle['policy_state_sha256'] == POLICY_STATE_SHA
                    and bundle['source'] == release['source'] and bundle['fixture_identity'] == release['fixture_identity']
                    and bundle['strict_loader']['probe_actions_byte_exact'] is True and bundle['strict_loader']['probe_rows'] == 32,
                    'C36 actor bundle identity/proof differs')
            with np.load(output/'actor.npz', allow_pickle=False) as archive:weights = {k:archive[k] for k in archive.files}
            validate_weights(weights, np)
            started = time.perf_counter()
            clipped, gained = [], []
            for row in inputs:
                attempted += 1
                a, b = numpy_action(row, weights, np)
                returned += 1
                clipped.append(a); gained.append(b)
            common_end = time.perf_counter()
            values = {'clipped': np.asarray(clipped, dtype=np.float32), 'gain0p5': np.asarray(gained, dtype=np.float32)}
            report.update(actual_nn_counts={'predict_calls': returned, 'predict_rows': returned, 'ppo_load': 0, 'torch_load': 0},
                          common192_predict_wall_s=common_end-started, cold_to_workload_end_s=common_end-_COLD_STARTED,
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          versions={'python': platform.python_version(), 'numpy': np.__version__},
                          forbidden_modules_absent=not any(n.partition('.')[0] in forbidden for n in sys.modules))
            require(attempted == returned == 192 and report['forbidden_modules_absent'] is True
                    and all(report['versions'][key] == bundle['versions'][key] for key in report['versions']), 'C36 NumPy counts/dependency boundary differs')
            with np.load(output/'torch_outputs.npz', allow_pickle=False) as archive:reference = {k:archive[k] for k in archive.files}
            report['parity_max_absolute_error'] = numerical_parity(reference, values, np)
            report.update(actor_manifest_identity=identity(output/'actor_manifest.json'), actor_identity=identity(output/'actor.npz'))
        require(inputs.tobytes() == original_input, 'C36 predictor mutated fixed inputs')
        require(not any(n.partition('.')[0] in forbidden for n in sys.modules), 'C36 forbidden dependency loaded')
        report['dependency_origins'] = dependency_origins(release['inputs'], {'numpy'} if stage == 'numpy' else {'numpy', 'torch', 'stable_baselines3', 'gymnasium'})
        with (output/(stage+'_outputs.npz')).open('xb') as stream:np.savez(stream, **values)
        report.update(status='COMPLETE', outputs_identity=identity(output/(stage+'_outputs.npz')))
    except BaseException as exc:  # noqa: BLE001 -- retain interrupted strict-load/counter evidence
        report['error'] = type(exc).__name__+': '+str(exc)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGXCPU):signal.signal(sig, signal.SIG_IGN)
        if counter is not None:report['known_actual_nn_counts'] = counter.summary()
        elif stage == 'numpy':report['known_actual_nn_counts'] = {'attempted': attempted, 'returned': returned}
        if values is not None and not (output/(stage+'_outputs.npz')).exists():
            try:
                with (output/(stage+'_outputs.npz')).open('xb') as stream:np.savez(stream, **values)
                report['outputs_identity'] = identity(output/(stage+'_outputs.npz'))
            except BaseException as exc:  # noqa: BLE001 -- preserve first numerical failure, record failed evidence write
                report.update(status='INCOMPLETE', error=(report['error'] or '')+'; output evidence: '+str(exc))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(worker_elapsed_s=time.perf_counter()-_COLD_STARTED, peak_rss_kib=usage.ru_maxrss,
                      cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime)
        write_json(output/(stage+'_report.json'), report)
    return report


def verify_worker(release, release_path, stage, value):
    require(value.get('schema') == 'b22-numpy36-worker-v1' and value.get('status') == 'COMPLETE' and value.get('error') is None
            and value.get('stage') == stage and value.get('release_identity') == identity(release_path)
            and value.get('physics_steps') == value.get('environment_constructions') == 0
            and value.get('qualification_granted') is False, 'C36 complete worker identity required')
    output = Path(release['plan']['output'])
    require(value['actor_identity'] == identity(output/'actor.npz')
            and value['actor_manifest_identity'] == identity(output/'actor_manifest.json')
            and value['outputs_identity'] == identity(output/(stage+'_outputs.npz')), 'C36 output identity differs')
    expected = {'ppo_load': 1, 'torch_load': 3, 'predict_calls': 193, 'predict_rows': 224} if stage == 'torch' else {'ppo_load': 0, 'torch_load': 0, 'predict_calls': 192, 'predict_rows': 192}
    require(value.get('actual_nn_counts') == expected and all(type(v) is int for v in value['actual_nn_counts'].values()), 'C36 exact NN budget differs')
    require(all(type(value.get(k)) in (int, float) and math.isfinite(value[k]) and value[k] >= 0
                for k in ('common192_predict_wall_s', 'cold_to_workload_end_s', 'peak_rss_kib'))
            and value['common192_predict_wall_s'] <= value['cold_to_workload_end_s'] <= LIMITS['worker_wall_s'], 'C36 timing/RSS missing or impossible')
    require(value['cold_to_workload_end_s'] > 0 and value['peak_rss_kib'] > 0, 'C36 cold wall/RSS must be measured positive values')
    if stage == 'numpy':
        require(value.get('forbidden_modules_absent') is True and set(value['parity_max_absolute_error']) == {'clipped', 'gain0p5'}
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= ABSOLUTE_LIMIT for v in value['parity_max_absolute_error'].values()), 'C36 NumPy parity/dependency check failed')
        require(value['parity_max_absolute_error'] == saved_parity(output), 'C36 claimed numerical maximum differs')
    else:
        proof, ledger = value['strict_loader'], value['api_ledger']
        require(proof['probe_rows'] == 32 and proof['probe_actions_byte_exact'] is True
                and proof['loaded_policy_state_sha256'] == POLICY_STATE_SHA and proof['metadata_sha256'] == METADATA_SHA
                and all(ledger['counts'].get(k, {}).get('attempted', 0) == ledger['counts'].get(k, {}).get('returned', 0) == v for k,v in API_LIMITS.items())
                and ledger['counts']['predict']['rows_returned'] == 224 and ledger['critic_rows'] == 0,
                'C36 strict loader/API ledger differs')


def execute(output):
    output = Path(output).resolve()
    require(ROOT == REGISTERED_ROOT and output.parent == ROOT/'results' and output.name.startswith('b22_numpy36_')
            and not output.exists() and not output.is_symlink(), 'C36 new registered output only; no retry')
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS['parent_cpu_soft_s'], LIMITS['parent_cpu_hard_s']))
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'C36 requires external Nice=10 or lower priority')
    config = read_json(ROOT/MANIFEST)
    require(identity(ROOT/MANIFEST)['sha256'] == MANIFEST_SHA, 'fixed v27 identity changed')
    for name, digest in PINS.items():require(identity(ROOT/name)['sha256'] == digest, 'C36 pinned helper changed: '+name)
    common = load_pinned('numpy36_supervision31', ROOT/'scripts/run_b22_suite31.py',
                         {'bytes': (ROOT/'scripts/run_b22_suite31.py').stat().st_size, 'sha256': PINS['scripts/run_b22_suite31.py']})
    require(common.memory() >= 6 << 30, 'NOT_READY: C36 startup RAM below 6 GiB')
    source = source_identity()
    inputs = {**config['source_hashes'], **source['authored'], **{str(ROOT/name): entry for name, entry in HISTORY.items()},
              str(ROOT/MANIFEST): identity(ROOT/MANIFEST),
              **{str(ROOT/name): {'bytes': (ROOT/name).stat().st_size, 'sha256': digest} for name, digest in PINS.items()}}
    payload, origins = fixed_inputs(config)
    require({'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()} == FIXTURE, 'predeclared192 input bytes differ')
    output.mkdir(exist_ok=False)
    with (output/'inputs.f32.bin').open('xb') as stream:stream.write(payload)
    write_json(output/'input_origins.json', {'rows': origins, 'recipe': plan()['fixture_recipe']})
    release = {'schema': 'b22-numpy36-release-v1', 'plan': plan(output), 'source': source, 'inputs': inputs,
               'fixture_identity': identity(output/'inputs.f32.bin'), 'origins_identity': identity(output/'input_origins.json')}
    release_path = output/'release.json'
    write_json(release_path, release)
    report = {'schema': 'b22-numpy36-report-v1', 'status': 'RUNNING', 'error': None, 'stages': {}, 'active_stage': None,
              'supervisor_pid': os.getpid(), 'release_identity': identity(release_path), 'actual_nn_counts': None,
              'qualification_granted': False, 'physics_steps': 0, 'environment_constructions': 0, 'retry_permitted': False}
    host, previous, deadline = None, {}, started+170
    def journal():
        temporary = output/'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'report.json')
    def stopped(signum, frame):
        raise InterruptedError('C36 parent interrupted: '+signal.Signals(signum).name)
    def sample(admission=False):
        available = common.memory()
        with (output/'resources.jsonl').open('a') as stream:stream.write(json.dumps({'effective_available_bytes': available, 'elapsed_s':time.monotonic()-started})+'\n')
        report['minimum_effective_bytes'] = min(report.get('minimum_effective_bytes', available), available)
        require(available >= (6 << 30 if admission else 4 << 30) and time.monotonic() < deadline, 'C36 RAM/wall reserve exhausted')
    journal()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGXCPU):previous[sig] = signal.signal(sig, stopped)
    alarm = signal.signal(signal.SIGALRM, stopped)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.monotonic()))
    try:
        check_release(release_path)
        host = load_pinned('numpy36_cleanup_host', Path(config['host']), config['source_hashes'][config['host']])
        host.subreaper()
        children = {}
        for stage in ('torch', 'numpy'):
            sample(True)
            check_release(release_path)
            for old, value in children.items():
                require(report['stages'][old]['report_identity'] == identity(output/(old+'_report.json')), 'C36 previous report changed')
                verify_worker(release, release_path, old, value)
            command = worker_argv(release_path, stage)
            nonce = os.urandom(24).hex()
            report['active_stage'] = stage
            report['stages'][stage] = {'status': 'STARTED', 'nonce': nonce, 'command': command}
            journal()
            env = dict(os.environ, B22_NUMPY36_NONCE=nonce, PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=config['runtime_environment']['PYTHONPATH'])
            for name in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'DISPLAY', 'XAUTHORITY'):env.pop(name, None)
            stage_started = time.monotonic()
            def spawn(argv, _env=env, **kwargs):return subprocess.Popen(argv, env=_env, cwd=ROOT, **kwargs)
            code = common.supervise(command, output/(stage+'_stdout.log'), host=host, sample=sample,
                                    limit=LIMITS['worker_wall_s'], deadline=deadline, factory=spawn)
            require(code == 0, 'C36 child failed: '+stage)
            child = read_json(output/(stage+'_report.json'))
            verify_worker(release, release_path, stage, child)
            children[stage] = child
            report['stages'][stage].update(status='COMPLETE', report_identity=identity(output/(stage+'_report.json')),
                                           process_wall_s=time.monotonic()-stage_started, actual_nn_counts=child['actual_nn_counts'])
            report['active_stage'] = None
            journal()
        check_release(release_path)
        require(source_identity() == source, 'C36 source revision changed')
        for stage in children:require(report['stages'][stage]['report_identity'] == identity(output/(stage+'_report.json')), 'C36 earlier report changed')
        for stage, value in children.items():verify_worker(release, release_path, stage, value)
        sample(False)
        report.update(status='COMPLETE', actual_nn_counts={'ppo_load':1, 'torch_load':3, 'torch_predict_calls':193, 'torch_predict_rows':224,
                                                         'numpy_predict_calls':192, 'numpy_predict_rows':192},
                      parity_max_absolute_error=children['numpy']['parity_max_absolute_error'],
                      measured_cold_to_workload_wall_ratio=children['numpy']['cold_to_workload_end_s']/children['torch']['cold_to_workload_end_s'],
                      measured_peak_rss_ratio=children['numpy']['peak_rss_kib']/children['torch']['peak_rss_kib'],
                      single_cold_pair_only=True,
                      measurement_scope='cold import/load plus common192; Torch includes mandatory original32 strict probe; not pure NN speed or control qualification')
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        report.update(status='INCOMPLETE', error=type(exc).__name__+': '+str(exc))
        active = report['active_stage']
        if active is not None:
            report['stages'][active]['status'] = 'INCOMPLETE'
            try:
                partial = read_json(output/(active+'_report.json'))
                require(partial['stage'] == active and partial['release_identity'] == identity(release_path), 'foreign C36 partial report')
                report['stages'][active]['known_nn_prefix'] = partial.get('known_actual_nn_counts')
            except (OSError, ValueError, KeyError, TypeError):
                report['stages'][active]['known_nn_prefix'] = None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, alarm)
        for sig in previous:signal.signal(sig, signal.SIG_IGN)
        if host is not None:
            try:
                report['cleanup'] = host.clean_descendants(os.getpid(), None)
                if report['cleanup']['remaining']:report.update(status='INCOMPLETE', error='C36 owned descendants remain')
            except (OSError, subprocess.SubprocessError) as exc:report.update(status='INCOMPLETE', error='C36 cleanup: '+str(exc))
        report['elapsed_s'] = time.monotonic()-started
        if report['elapsed_s'] >= LIMITS['total_wall_s']:report.update(status='INCOMPLETE', error='C36 final wall exhausted')
        if report['status'] != 'COMPLETE':report['actual_nn_counts'] = None
        for sig, handler in previous.items():signal.signal(sig, handler)
        journal()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=('torch', 'numpy'))
    parser.add_argument('--release', type=Path)
    args = parser.parse_args(argv)
    require(not (args.execute and args.worker), 'choose one C36 role')
    if args.worker:
        require(args.release is not None, 'supervised C36 release required')
        value = worker(args.release.resolve(), args.worker)
    elif args.execute:
        require(args.output is not None, 'explicit new output required')
        value = execute(args.output)
    else:value = plan(args.output)
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if value['status'] in ('COMPLETE', 'PLAN_ONLY') else 1


if __name__ == '__main__':
    raise SystemExit(main())
