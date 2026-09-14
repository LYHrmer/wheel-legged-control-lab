"""One fixed training-terrain check of final saved policies and zero residual.

No training, checkpoint selection, retry, GUI, or repository-source writes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=serializable)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    sys.path.insert(0, str(args.repo.resolve()/'src'))
    import torch
    from scripts.run_d1_course_curriculum import forward_command, source_hashes
    from scripts.d1_course_curriculum import scaled_training_terrain
    from wheel_legged_control.d1.state_provider import D1StateProviderConfig
    from wheel_legged_control.d1.locomotion_checkpoint import load_locomotion_policy
    from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv

    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    terrain = scaled_training_terrain(3, 0)
    policies = ['zero', 'flat', 'mixed', 'curriculum']
    model_files = {str(args.models/name/file): digest(args.models/name/file)
                   for name in policies[1:] for file in ('model.zip', 'model.metadata.json')}
    sources = source_hashes()
    protocol = {
        'schema': 'd1-course-final-policy-development-check-v1',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'policies_in_run_order': policies,
        'seed': 55101, 'episode_seconds': 32.0, 'max_transitions': 3200,
        'baseline': 'wheel_leg', 'action_mode': 'independent8', 'state_source': 'oracle',
        'terrain': asdict(terrain), 'terrain_definition': 'scaled_training_terrain(3, 0)',
        'command_source': 'scripts.run_d1_course_curriculum.forward_command',
        'command_definition': '0 m/s through .5s, linear ramp to .25m/s by 1s, then .25m/s; yaw 0, clearance .455m',
        'deterministic_policy': True,
        'checkpoint_selection': 'all three supplied final checkpoints; no selection',
        'classification': 'single training-terrain development check; not holdout or training-episode comparison',
        'stop_rule': 'first terminated or truncated, no reset/retry in an episode',
        'model_sha256': model_files, 'source_sha256': sources,
        'helper_sha256': digest(__file__),
        'versions': {name: version(name) for name in ('numpy', 'mujoco', 'gymnasium', 'stable-baselines3', 'torch')},
    }
    write(output/'protocol.json', protocol)
    summaries = {}
    initial_reference = None
    for name in policies:
        case = output/name
        case.mkdir()
        env = D1LocomotionEnv(baseline='wheel_leg', action_mode='independent8',
                              provider_config=D1StateProviderConfig('oracle'),
                              terrain=terrain, episode_seconds=32.0,
                              command_source=forward_command)
        rows, obs_history, qpos, qvel = [], [], [], []
        states_time = []
        started = time.monotonic()
        try:
            obs, initial_info = env.reset(seed=55101)
            initial = {'episode_metadata': initial_info['episode_metadata'],
                       'observation_sha256': array_digest(obs),
                       'qpos_sha256': array_digest(env.plant.data.qpos),
                       'qvel_sha256': array_digest(env.plant.data.qvel)}
            write(case/'initial.json', initial)
            if initial_reference is None:
                initial_reference = initial
            elif initial != initial_reference:
                raise RuntimeError('initial state/observation/metadata differs across policies')
            model = (None if name == 'zero' else load_locomotion_policy(
                args.models/name/'model.zip', args.models/name/'model.metadata.json', env))
            initial_x = float(env.plant.data.qpos[0])
            obs_history.append(obs.copy())
            qpos.append(env.plant.data.qpos.copy())
            qvel.append(env.plant.data.qvel.copy())
            states_time.append(float(env.plant.data.time))
            with (case/'trace.jsonl').open('x') as stream:
                while True:
                    action = (np.zeros(8, dtype=np.float32) if model is None
                              else model.predict(obs, deterministic=True)[0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    if not np.isfinite(obs).all() or not np.isfinite(reward):
                        raise RuntimeError('nonfinite observation or reward')
                    row = {'transition': len(rows)+1, 'reward': reward,
                           'terminated': bool(terminated), 'truncated': bool(truncated),
                           'actor_action': np.asarray(action).copy(), **info}
                    # Preserve the complete metrics, reward terms, command, exposure,
                    # and every policy/raw/applied action rather than averages only.
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False,
                                            default=serializable)+'\n')
                    stream.flush()
                    rows.append(row)
                    obs_history.append(obs.copy())
                    qpos.append(env.plant.data.qpos.copy())
                    qvel.append(env.plant.data.qvel.copy())
                    states_time.append(float(env.plant.data.time))
                    if terminated or truncated:
                        break
            rmse = lambda field: float(np.sqrt(np.mean([r['metrics'][field]**2 for r in rows])))
            summary = {
                'policy': name, 'actual_transitions': len(rows),
                'terminal_reason': rows[-1]['terminal_reason'],
                'terminated': rows[-1]['terminated'], 'truncated': rows[-1]['truncated'],
                'integrated_duration_s': float(env.plant.data.time),
                'initial_x_m': initial_x, 'final_x_m': rows[-1]['metrics']['x_m'],
                'signed_forward_progress_m': rows[-1]['metrics']['x_m']-initial_x,
                'velocity_rmse_mps': rmse('velocity_error_mps'),
                'yaw_rate_rmse_rps': rmse('yaw_rate_error_rps'),
                'height_rmse_m': rmse('height_error_m'),
                'cumulative_return': float(sum(r['reward'] for r in rows)),
                'cumulative_reward_terms': {key: float(sum(r['reward_terms'][key] for r in rows))
                                            for key in rows[0]['reward_terms']},
                'mean_torque_input_clipped_fraction': float(np.mean(
                    [r['metrics']['torque_input_clipped_fraction'] for r in rows])),
                'actor_action_at_bound_fraction': float(np.mean(
                    np.abs([r['actor_action'] for r in rows]) >= .999)),
                'geometric_nonflat_exposure': rows[-1]['terrain_exposure'],
                'minimum_clearance_m': min(r['metrics']['clearance_m'] for r in rows),
                'maximum_undesired_ground_contacts': max(r['metrics']['undesired_ground_contacts'] for r in rows),
                'initial_state_observation_metadata_same': initial == initial_reference,
                'wall_seconds': time.monotonic()-started,
            }
            write(case/'summary.json', summary)
            summaries[name] = summary
            print(json.dumps(summary, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {'exception': type(exc).__name__, 'message': str(exc),
                       'completed_transitions': len(rows)}
            write(case/'failure.json', failure)
            summaries[name] = failure
        finally:
            with (case/'states.npz').open('xb') as stream:
                np.savez_compressed(stream, observation=np.asarray(obs_history),
                                    qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                                    time_s=np.asarray(states_time))
            env.close()
            write(case/'manifest.json', {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                                          for p in sorted(case.iterdir()) if p.is_file()})
    unchanged = source_hashes() == sources and all(digest(path) == value for path, value in model_files.items())
    report = {'schema': protocol['schema'], 'summaries': summaries,
              'source_and_models_unchanged': unchanged,
              'all_four_completed': all('actual_transitions' in s for s in summaries.values()),
              'classification': protocol['classification']}
    write(output/'report.json', report)
    write(output/'manifest.json', {str(p.relative_to(output)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                                   for p in sorted(output.rglob('*')) if p.is_file()})
    return 0 if unchanged and report['all_four_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
