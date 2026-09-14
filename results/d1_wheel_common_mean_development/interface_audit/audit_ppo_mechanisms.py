"""Read-only PPO timing/model audit using recorded observations; no physics or training."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--development', required=True, type=Path)
    parser.add_argument('--final-check', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo/'src'))
    from scripts.run_d1_locomotion_experiment import PPO_SETTINGS
    from scripts.run_d1_course_curriculum import source_hashes
    from stable_baselines3 import PPO
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    sources = source_hashes()
    original_models = {str(args.development/c/f): sha(args.development/c/f)
                       for c in ('flat', 'mixed', 'curriculum')
                       for f in ('model.zip', 'model.metadata.json', 'progress.csv', 'episodes.json')}
    input_manifest = json.loads((args.final_check/'manifest.json').read_text())
    assert all(sha(args.final_check/p) == entry['sha256'] for p, entry in input_manifest.items())
    write(args.output/'protocol.json', {
        'scope': 'model inference and replay of recorded data only',
        'physics_steps': 0, 'training_updates': 0,
        'repo_source_sha256': sources, 'training_files_sha256': original_models,
        'input_manifest_sha256': sha(args.final_check/'manifest.json'),
        'helper_sha256': sha(__file__),
        'checks': ['actor replay from obs[k]', 'previous action obs[k+1]', 'command in obs[k]',
                   'critic and training-log statistics', 'fresh initialized policy versus saved models',
                   'actual SB3 timeout plumbing using synthetic terminal replay, no optimizer call'],
    })

    class SpacesOnly(gym.Env):
        observation_space = gym.spaces.Box(-5, 5, (82,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (8,), np.float32)

        def reset(self, *, seed=None, options=None):
            raise AssertionError('fresh model construction must not reset or run an environment')

        def step(self, action):
            raise AssertionError('no environment steps permitted for fresh model')

    fresh = PPO('MlpPolicy', SpacesOnly(), seed=47000, device='cpu', verbose=0, **PPO_SETTINGS)
    with np.load(args.final_check/'zero/states.npz', allow_pickle=False) as source:
        common_obs = source['observation'].copy()
    fresh_mean = fresh.predict(common_obs, deterministic=True)[0]
    fresh_record = {'construction_only': True, 'seed': 47000,
                    'gradient_updates': fresh._n_updates, 'num_timesteps': fresh.num_timesteps,
                    'mean_action_rms_on_zero_reference': float(np.sqrt(np.mean(fresh_mean**2))),
                    'max_abs_mean_action_on_zero_reference': float(np.abs(fresh_mean).max()),
                    'first_observation_deterministic_action': fresh_mean[0].tolist(),
                    'std': fresh.policy.log_std.exp().detach().cpu().numpy().tolist()}
    reports = {}
    for name in ('zero', 'flat', 'mixed', 'curriculum'):
        case = args.final_check/name
        with (case/'trace.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        with np.load(case/'states.npz', allow_pickle=False) as source:
            obs = source['observation'].copy()
        actions = np.asarray([r['actor_action'] for r in rows], dtype=np.float32)
        applied = np.asarray([r['applied_action'] for r in rows], dtype=np.float32)
        assert np.array_equal(obs[0, 74:82], np.zeros(8, np.float32))
        previous_error = float(np.abs(obs[1:, 74:82]-applied).max())
        assert previous_error == 0
        command = np.asarray([r['command']['forward_velocity_mps']/.6 for r in rows], dtype=np.float32)
        assert np.array_equal(obs[:-1, 38], command)
        assert all(r['policy_action'] == r['raw_action'] == r['applied_action'] for r in rows)
        report = {'recorded_steps': len(rows), 'previous_action_slice_max_error': previous_error,
                  'command_slice_exact': True, 'policy_raw_applied_actions_identical': True,
                  'reward_per_tick_mean': float(np.mean([r['reward'] for r in rows])),
                  'reward_per_tick_min': min(r['reward'] for r in rows),
                  'reward_per_tick_max': max(r['reward'] for r in rows)}
        if name != 'zero':
            model = PPO.load(args.development/name/'model.zip', device='cpu')
            model.policy.set_training_mode(False)
            inferred = model.predict(obs[:-1], deterministic=True)[0]
            error = float(np.abs(inferred-actions).max())
            assert error < 2e-6, error
            with torch.no_grad():
                values = model.policy.predict_values(torch.as_tensor(obs)).reshape(-1).cpu().numpy()
            rewards = np.asarray([r['reward'] for r in rows])
            td = rewards + model.gamma*values[1:] - values[:-1]
            mc = np.empty(len(rewards))
            value = float(values[-1])  # Administrative time limit bootstraps terminal V.
            for index in range(len(rewards)-1, -1, -1):
                value = float(rewards[index]) + model.gamma*value
                mc[index] = value
            report.update({
                'actor_inference_max_abs_error': error,
                'checkpoint_actual_updates': int(model._n_updates),
                'checkpoint_actual_timesteps': int(model.num_timesteps),
                'gamma': model.gamma, 'gae_lambda': model.gae_lambda,
                'normalize_advantage': bool(model.normalize_advantage),
                'vf_coef': float(model.vf_coef),
                'clip_range_vf': None if model.clip_range_vf is None else 'callable',
                'target_kl': model.target_kl,
                'mean_action_rms_on_common_zero_obs': float(np.sqrt(np.mean(model.predict(common_obs, deterministic=True)[0]**2))),
                'first_observation_deterministic_action': model.predict(obs[0], deterministic=True)[0].tolist(),
                'std': model.policy.log_std.exp().detach().cpu().numpy().tolist(),
                'critic_value_mean_on_own_deterministic_trace': float(values.mean()),
                'critic_value_min': float(values.min()), 'critic_value_max': float(values.max()),
                'own_trace_one_step_td_rmse': float(np.sqrt(np.mean(td**2))),
                'own_trace_discounted_bootstrapped_return_rmse': float(np.sqrt(np.mean((mc-values[:-1])**2))),
                'critic_diagnostic_limit': 'Deterministic evaluation trace; not reconstructed stochastic training targets or training advantage values.',
                'policy_parameter_delta_l2_from_seed_reconstruction': float(math.sqrt(sum(
                    float(torch.sum((value-fresh.policy.state_dict()[key])**2))
                    for key, value in model.policy.state_dict().items()))),
            })
            with (args.development/name/'progress.csv').open() as stream:
                logs = list(csv.DictReader(stream))
            log_stats = {}
            for field in ('train/value_loss', 'train/explained_variance', 'train/approx_kl',
                          'train/clip_fraction', 'train/std'):
                vals = np.asarray([float(row[field]) for row in logs if row.get(field)], dtype=float)
                assert np.isfinite(vals).all()
                log_stats[field] = {'count': len(vals), 'min': float(vals.min()),
                                    'max': float(vals.max()), 'last': float(vals[-1]),
                                    'mean': float(vals.mean())}
            ev = [float(row['train/explained_variance']) for row in logs if row.get('train/explained_variance')]
            log_stats['negative_explained_variance_count'] = sum(value < 0 for value in ev)
            log_stats['last_logged_updates'] = int(logs[-1]['train/n_updates'])
            report['training_log_statistics'] = log_stats
            episodes = json.loads((args.development/name/'episodes.json').read_text())
            report['episode_exposure'] = [{'level': r['level'], 'terrain_index': r['terrain_index'],
                'steps': r['transitions'], 'nonflat_steps': (r['last_terrain_exposure'] or {}).get('nonflat_steps', 0),
                'terminal_reason': r['terminal_reason']} for r in episodes]
        reports[name] = report

    # Feed one preserved final transition through the installed SB3 machinery.
    # Replayed observations ignore the newly sampled action; this checks plumbing,
    # not physics or policy performance. No learn()/train()/optimizer calls occur.
    with np.load(args.final_check/'flat/states.npz', allow_pickle=False) as source:
        before, after = source['observation'][-2:].copy()
    with (args.final_check/'flat/trace.jsonl').open() as stream:
        last_row = json.loads(list(stream)[-1])
    raw_reward = last_row['reward']

    class TerminalReplay(gym.Env):
        observation_space = SpacesOnly.observation_space
        action_space = SpacesOnly.action_space

        def __init__(self, truncated):
            self.truncated = truncated
            self.calls = Counter()

        def reset(self, *, seed=None, options=None):
            self.calls['reset'] += 1
            return before.copy(), {}

        def step(self, action):
            self.calls['replayed_step'] += 1
            return after.copy(), raw_reward, not self.truncated, self.truncated, {}

    bootstrap = []
    for truncation in (True, False):
        replay = TerminalReplay(truncation)
        vec = DummyVecEnv([lambda: replay])
        model = PPO.load(args.development/'flat/model.zip', env=vec, device='cpu')
        before_parameters = {name: value.clone() for name, value in model.policy.state_dict().items()}
        old_updates = model._n_updates
        model.rollout_buffer = RolloutBuffer(1, model.observation_space, model.action_space,
                                            device='cpu', gamma=model.gamma, gae_lambda=model.gae_lambda)
        _, callback = model._setup_learn(1, reset_num_timesteps=False)
        model.collect_rollouts(vec, callback, model.rollout_buffer, n_rollout_steps=1)
        with torch.no_grad():
            terminal_value = float(model.policy.predict_values(torch.as_tensor(after[None, :])).reshape(-1)[0])
        expected = raw_reward + (model.gamma*terminal_value if truncation else 0.)
        actual = float(model.rollout_buffer.rewards[0, 0])
        assert abs(actual-expected) < 2e-6
        assert old_updates == model._n_updates
        assert all(torch.equal(value, before_parameters[name]) for name, value in model.policy.state_dict().items())
        bootstrap.append({'truncated': truncation, 'raw_recorded_reward': raw_reward,
                          'terminal_value': terminal_value, 'expected_buffer_reward': expected,
                          'actual_buffer_reward': actual, 'parameter_updates': 0,
                          'replay_calls': dict(replay.calls), 'physics_steps': 0,
                          'terminal_observation_used_instead_of_reset_observation': bool(truncation)})
        vec.close()

    assert source_hashes() == sources
    assert all(sha(path) == value for path, value in original_models.items())
    report = {'status': 'passed', 'fresh_policy': fresh_record, 'policies': reports,
              'timeout_bootstrap_replay': bootstrap,
              'discount_efold_seconds': -.01/math.log(PPO_SETTINGS['gamma']),
              'gae_efold_seconds': -.01/math.log(PPO_SETTINGS['gamma']*PPO_SETTINGS['gae_lambda']),
              'rollout_seconds': .01*PPO_SETTINGS['n_steps'],
              'physics_steps': 0, 'optimizer_updates': 0,
              'source_training_records_and_models_unchanged': True}
    write(args.output/'report.json', report)
    print(json.dumps({'status': report['status'], 'policies': reports,
                      'fresh_policy': fresh_record, 'bootstrap': bootstrap}, indent=2))


if __name__ == '__main__':
    main()
