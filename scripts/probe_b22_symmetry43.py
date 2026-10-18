"""One completed42-deficit8 actor export and mirror numerical probe; zero physics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import site
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in reversed((Path(site.getusersitepackages()), ROOT/'.local-deps', ROOT/'src', ROOT)):
    while str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
_guard_spec = importlib.util.spec_from_file_location('symmetry43_guard', ROOT/'scripts/resource_guard42.py')
_guard = importlib.util.module_from_spec(_guard_spec)
_guard_spec.loader.exec_module(_guard)
check_memory, snapshot_memory = _guard.check_memory, _guard.snapshot_memory

UNIT = 'wheel-symmetry43-numeric03-20261018-01.service'
SCOPE = f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{UNIT}'
OUTPUT = ROOT/'results/b22_symmetry43_numeric_20261018_03'
PLAN = ROOT/'results/b22_symmetry43_numeric_plan_20261018_03.json'
FOLDER = ROOT/'results/b22_tracking_reward42_20261018_05/train_deficit8/final_checkpoint'
FIXTURE = ROOT/'results/b22_numpy36_20261018_01/inputs.f32.bin'
MODEL_SHA = '44cbf9a4363ff42af78d8e54049e0dbade2bea74701e9a27073bcb9e656f3239'
FIXTURE_SHA = 'b1468a1d91007cecb3d8c8a706fb1f44fb2857819eb96f57fd9c72c5cfeda834'
AUTHORED = ('scripts/probe_b22_symmetry43.py', 'src/wheel_legged_control/d1/policy_symmetry43.py',
            'scripts/audit_b22_symmetry43_geometry.py', 'tests/test_policy_symmetry43.py',
            'tests/test_probe_b22_symmetry43.py', 'docs/b22_symmetry43.md',
            'scripts/export_b22_numpy36.py', 'scripts/resource_guard42.py')
WEIGHTS = {'w1': 'mlp_extractor.policy_net.0.weight', 'b1': 'mlp_extractor.policy_net.0.bias',
           'w2': 'mlp_extractor.policy_net.2.weight', 'b2': 'mlp_extractor.policy_net.2.bias',
           'w3': 'action_net.weight', 'b3': 'action_net.bias'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def inputs():
    require(sha(FOLDER/'final_model.zip') == MODEL_SHA and sha(FIXTURE) == FIXTURE_SHA,
            'fixed completed model/fixture differ')
    manifest = json.loads((FOLDER.parent/'final_checkpoint_manifest.json').read_text())
    for name, entry in manifest['files'].items():
        require(sha(FOLDER/name) == entry['sha256'] and (FOLDER/name).stat().st_size == entry['bytes'],
                'final42 file differs')
    require(sha(FOLDER/'final_metadata.json') == manifest['metadata_sha256'], 'metadata differs')
    require(json.loads((FOLDER.parent.parent/'report.json').read_text())['status'] == 'COMPLETE',
            'completed42 only')
    paths = [FIXTURE, FOLDER.parent/'final_checkpoint_manifest.json', FOLDER.parent.parent/'report.json',
             FOLDER/'final_metadata.json', *(FOLDER/name for name in manifest['files'])]
    return {str(p): sha(p) for p in paths}


def sources(proof=None):
    if proof is None:
        proof = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'files': {name: sha(ROOT/name) for name in AUTHORED}}
    require(set(proof['files']) == set(AUTHORED), 'source closure differs')
    for name, digest in proof['files'].items():
        require(sha(ROOT/name) == digest, 'source bytes differ')
        committed = subprocess.check_output(['git', 'show', proof['commit']+':'+name], cwd=ROOT, timeout=10)
        require(hashlib.sha256(committed).hexdigest() == digest, 'commit source first')
    return proof


def scope():
    return {'schema': 'b22-symmetry43-numeric-plan-v1', 'output': str(OUTPUT), 'unit': UNIT,
            'base_checkpoint_sha256': MODEL_SHA, 'fixture_sha256': FIXTURE_SHA,
            'fixture_rows': 192, 'fixture_recipe': 'own42_probe32 + first160_of36; each and its mirror',
            'ppo_load_max': 1, 'torch_load_max': 3, 'sb3_predict_max': 384, 'numpy_predict_max': 384,
            'symmetric_predict_max': 192, 'physics_steps': 0, 'training_calls': 0,
            'environment_constructions': 0, 'wall_s': 180, 'admission_stable_s': 10,
            'start_effective_bytes': 4 << 30, 'reserve_effective_bytes': 2 << 30,
            'memory_high_bytes': 2 << 30, 'memory_max_bytes': 3 << 30, 'swap_max_bytes': 0,
            'absolute_tolerance': 1e-6, 'retry_permitted': False, 'closed_loop_qualification': False}


def load_symmetry():
    # Package __init__ eagerly imports the simulator; this pure numerical path
    # deliberately loads the same SHA-bound module without that package chain.
    spec = importlib.util.spec_from_file_location('symmetry43_pure', ROOT/AUTHORED[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract(model, np, torch):
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import FlattenExtractor
    p = model.policy
    require(type(p) is ActorCriticPolicy and type(p.action_dist) is DiagGaussianDistribution and
            type(p.features_extractor) is FlattenExtractor and p.share_features_extractor and
            p.pi_features_extractor is p.vf_features_extractor is p.features_extractor and
            not list(p.features_extractor.parameters()) and not p.squash_output and not p.use_sde and
            model.get_env() is None and model.get_vec_normalize_env() is None and str(model.device) == 'cpu',
            'stateless unsquashed mean actor required')
    for space, size, limit in ((model.observation_space, 99, 5), (model.action_space, 16, 1)):
        require(space.shape == (size,) and space.dtype == np.float32 and
                (space.low == -limit).all() and (space.high == limit).all(), 'wrong actor spaces')
    flatten = list(p.features_extractor.children())
    require(len(flatten) == 1 and type(flatten[0]) is torch.nn.Flatten and
            flatten[0].start_dim == 1 and flatten[0].end_dim == -1, 'unexpected feature graph')
    stack = p.mlp_extractor.policy_net
    require(type(stack) is torch.nn.Sequential and tuple(type(x) for x in stack) ==
            (torch.nn.Linear, torch.nn.Tanh, torch.nn.Linear, torch.nn.Tanh) and
            (stack[0].in_features, stack[0].out_features, stack[2].in_features, stack[2].out_features) ==
            (99, 128, 128, 128) and type(p.action_net) is torch.nn.Linear and
            (p.action_net.in_features, p.action_net.out_features) == (128, 16), 'unexpected actor graph')
    state = p.state_dict()
    require({k for k in state if k.startswith(('mlp_extractor.policy_net.', 'action_net.'))} == set(WEIGHTS.values()),
            'extra actor state')
    require(all(state[k].dtype == torch.float32 and state[k].device.type == 'cpu' for k in WEIGHTS.values()),
            'wrong actor tensor device/type')
    return {name: state[key].detach().numpy().copy() for name, key in WEIGHTS.items()}


def compare(reference, actual, np):
    require(reference.shape == actual.shape == (384, 16) and reference.dtype == actual.dtype == np.float32
            and np.isfinite(reference).all() and np.isfinite(actual).all(), 'invalid parity arrays')
    maximum = float(np.max(np.abs(reference.astype(np.float64)-actual.astype(np.float64))))
    require(maximum <= 1e-6, 'new42 NumPy actor exceeds original numerical tolerance')
    return maximum


def execute():
    plan = json.loads(PLAN.read_text())
    require(plan['scope'] == scope() and plan['inputs'] == inputs(), 'fixed numeric plan differs')
    sources(plan['source'])
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Nice10 required')
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'INCOMPLETE', 'error': None, 'plan_sha256': sha(PLAN), 'source': plan['source'],
              'ppo_load_attempts': 0, 'ppo_load_returns': 0, 'torch_load_attempts': 0,
              'sb3_predict_attempts': 0, 'sb3_predict_returns': 0,
              'numpy_predict_attempts': 0, 'numpy_predict_returns': 0,
              'physics_steps': 0, 'environment_constructions': 0, 'training_calls': 0,
              'closed_loop_qualification': False}
    old_load = None
    def poll():
        require(time.monotonic()-started < 180, 'numeric wall limit')
        memory = snapshot_memory(SCOPE)
        with (OUTPUT/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(memory)+'\n')
        check_memory(memory)
        require(memory['effective_available_bytes'] >= 2 << 30, 'runtime reserve')
        return memory['effective_available_bytes']
    def interrupted(signum, frame):
        raise RuntimeError('received '+signal.Signals(signum).name)
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        stable = None
        while True:
            available, now = poll(), time.monotonic()
            stable = (now if stable is None else stable) if available >= 4 << 30 else None
            if stable is not None and now-stable >= 10:
                break
            time.sleep(1)
        spec = importlib.util.spec_from_file_location('numpy36_math43', ROOT/'scripts/export_b22_numpy36.py')
        math36 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(math36)
        math36.block_imports(math36.SIMULATORS)
        import numpy as np
        import stable_baselines3 as sb3
        import torch

        symmetry = load_symmetry()
        SymmetricActor = symmetry.SymmetricActor
        mirror_action, mirror_observation = symmetry.mirror_action, symmetry.mirror_observation
        metadata = json.loads((FOLDER/'final_metadata.json').read_text())
        report['versions'] = {'numpy': np.__version__, 'torch': torch.__version__, 'sb3': sb3.__version__}
        report['module_origins'] = {name: str(Path(module.__file__).resolve())
                                    for name, module in (('numpy', np), ('torch', torch), ('sb3', sb3))}
        require((np.__version__, torch.__version__, sb3.__version__) ==
                (metadata['numpy_version'], metadata['torch_version'], metadata['sb3_version']), 'runtime versions differ')
        for path in report['module_origins'].values():
            require(metadata['dependency_hashes'][path]['sha256'] == sha(path), 'training dependency root changed')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        old_load = torch.load
        def load(*args, **kwargs):
            report['torch_load_attempts'] += 1
            require(report['torch_load_attempts'] <= 3, 'torch load cap')
            return old_load(*args, **kwargs)
        torch.load = load
        report['ppo_load_attempts'] += 1
        model = sb3.PPO.load(FOLDER/'final_model.zip', device='cpu')
        report['ppo_load_returns'] += 1
        weights = extract(model, np, torch)
        math36.validate_weights(weights, np)
        original = np.fromfile(FOLDER/'final_probe_observations_f32.bin', dtype='<f4').reshape(32, 99)
        historical = np.fromfile(FIXTURE, dtype='<f4').reshape(192, 99)[:160]
        cohort = np.concatenate([original, historical])
        reference, actual, symmetric = [], [], []
        def predict_numpy(obs):
            report['numpy_predict_attempts'] += 1
            require(report['numpy_predict_attempts'] <= 384, 'NumPy call cap')
            action = math36.numpy_action(obs, weights, np)[0]
            actual.append(action.copy())
            report['numpy_predict_returns'] += 1
            return action
        projected = SymmetricActor(predict_numpy)
        for i, row in enumerate(cohort):
            if i % 16 == 0:
                poll()
            for obs in (row, mirror_observation(row)):
                report['sb3_predict_attempts'] += 1
                require(report['sb3_predict_attempts'] <= 384, 'SB3 predict cap')
                action, state = model.predict(obs, deterministic=True)
                require(state is None and action.shape == (16,) and action.dtype == np.float32, 'actor return differs')
                reference.append(action.copy())
                report['sb3_predict_returns'] += 1
            symmetric.append(projected(row))
        reference, actual, symmetric = map(np.asarray, (reference, actual, symmetric))
        report['max_absolute_error'] = compare(reference, actual, np)
        expected_probe = np.fromfile(FOLDER/'final_probe_actions_f32.bin', dtype='<f4').reshape(32, 16)
        probe_error = float(np.max(np.abs(reference[:64:2].astype(np.float64)-expected_probe)))
        require(probe_error <= 1e-6, 'complete42 original saved probe differs')
        report['original_probe32_max_error'] = probe_error
        sym_reference = np.asarray([(reference[i].astype(np.float64)+mirror_action(reference[i+1]))*.5
                                    for i in range(0, 384, 2)], dtype=np.float32)
        report['symmetric_max_absolute_error'] = float(np.max(np.abs(symmetric.astype(np.float64)-sym_reference)))
        require(report['symmetric_max_absolute_error'] <= 1e-6, 'symmetric parity differs')
        require(projected.actor_attempts == projected.actor_returns == 384 and projected.predict_calls == 192,
                'symmetric call counts differ')
        report['symmetric_predict_returns'] = projected.predict_calls
        np.savez(OUTPUT/'actor.npz', **weights)
        np.savez(OUTPUT/'numerical.npz', observations=cohort, sb3=reference, numpy=actual, symmetric=symmetric)
        report['artifacts'] = {name: sha(OUTPUT/name) for name in ('actor.npz', 'numerical.npz')}
        require(inputs() == plan['inputs'] and sha(PLAN) == report['plan_sha256'], 'inputs changed')
        sources(plan['source'])
        poll()
        report.update(status='COMPLETE', base_checkpoint_sha256=MODEL_SHA)
    except BaseException as exc:  # noqa: BLE001 -- preserve interruption and partial call counts
        report['error'] = f'{type(exc).__name__}: {exc}'
        (OUTPUT/'error.txt').write_text(traceback.format_exc())
    finally:
        for s in previous:
            signal.signal(s, signal.SIG_IGN)
        if old_load is not None:
            torch.load = old_load
        report['elapsed_s'] = time.monotonic()-started
        write(OUTPUT/'report.json', report)
        for s, handler in previous.items():
            signal.signal(s, handler)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = execute()
        print(json.dumps(result, indent=2))
        return int(result['status'] != 'COMPLETE')
    if args.prepare:
        require(not PLAN.exists() and not OUTPUT.exists(), 'one numeric attempt only')
        plan = {'scope': scope(), 'inputs': inputs(), 'source': sources()}
        with PLAN.open('x') as stream:
            json.dump(plan, stream, indent=2)
    print(json.dumps({'status': 'PREPARED' if args.prepare else 'PLAN_ONLY', **scope()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
