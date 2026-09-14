"""Closed-loop interventions on a fixed saved policy, with unchanged physics."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=6)
    p.add_argument('--terrain', choices=('flat', 'road'), default='flat')
    p.add_argument('--require-no-regression', action='store_true')
    a = p.parse_args()
    sys.path[:0] = [str(a.repo), str(a.repo/'src')]
    import torch
    from scripts.d1_course_curriculum import scaled_training_terrain
    from scripts.run_d1_course_curriculum import forward_command, source_hashes
    from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
    from wheel_legged_control.d1.locomotion_checkpoint import load_locomotion_policy
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    terrain = scaled_training_terrain(0 if a.terrain == 'flat' else 3, 0)
    sha = lambda x: hashlib.sha256(Path(x).read_bytes()).hexdigest()
    model_file, metadata = a.model_dir/'model.zip', a.model_dir/'model.metadata.json'
    source = source_hashes()
    modes = ['zero', 'full', 'legs_only', 'wheels_only', 'wheel_demean']
    protocol = {'modes': modes, 'seconds': a.seconds, 'seed': 55101,
                'terrain': asdict(terrain), 'model_sha256': sha(model_file),
                'metadata_sha256': sha(metadata), 'source_sha256': source,
                'command': 'forward_command: .5s stand, .5s ramp, .25m/s thereafter',
                'classification': 'development diagnostic; inference action interventions, not trained policies',
                'prediction': 'zeroing negative wheel residuals should reduce velocity error if they cause slowdown'}
    write(a.output/'protocol.json', protocol)
    results = {}
    initial_obs = None
    for mode in modes:
        env = D1LocomotionEnv(terrain=terrain, episode_seconds=a.seconds,
                              command_source=forward_command, action_mode='independent8')
        rows = []
        try:
            obs, _ = env.reset(seed=55101)
            if initial_obs is None:
                initial_obs = obs.copy()
            assert np.array_equal(obs, initial_obs)
            model = load_locomotion_policy(model_file, metadata, env)
            start_x = float(env.plant.data.qpos[0])
            while True:
                predicted = model.predict(obs, deterministic=True)[0]
                action = predicted.copy()
                if mode == 'zero':
                    action[:] = 0
                elif mode == 'legs_only':
                    action[4:] = 0
                elif mode == 'wheels_only':
                    action[:4] = 0
                elif mode == 'wheel_demean':
                    action[4:] -= action[4:].mean()
                obs, reward, term, trunc, info = env.step(action)
                rows.append({'metrics': info['metrics'], 'reward': reward,
                             'reward_terms': info['reward_terms'],
                             'prediction': predicted.tolist(), 'intervened_action': action.tolist(),
                             'applied_action': info['applied_action'].tolist(),
                             'terminal_reason': info['terminal_reason']})
                if term or trunc:
                    break
            rmse = lambda name: float(np.sqrt(np.mean([r['metrics'][name]**2 for r in rows])))
            result = {'steps': len(rows), 'reason': rows[-1]['terminal_reason'],
                      'progress_m': rows[-1]['metrics']['x_m']-start_x,
                      'velocity_rmse_mps': rmse('velocity_error_mps'),
                      'yaw_rmse_rps': rmse('yaw_rate_error_rps'),
                      'height_rmse_m': rmse('height_error_m'),
                      'return': float(sum(r['reward'] for r in rows)),
                      'reward_terms': {k: float(sum(r['reward_terms'][k] for r in rows))
                                       for k in rows[0]['reward_terms']},
                      'mean_executed_action': np.mean([r['applied_action'] for r in rows], axis=0).tolist()}
            write(a.output/f'{mode}.json', rows)
            results[mode] = result
            print(json.dumps({'mode': mode, **result}), flush=True)
        finally:
            env.close()
    assert source_hashes() == source
    assert sha(model_file) == protocol['model_sha256']
    regression = results['full']['velocity_rmse_mps'] > 1.1*results['zero']['velocity_rmse_mps']
    write(a.output/'summary.json', {'results': results, 'velocity_regression_reproduced': regression,
                                   'sources_and_checkpoint_unchanged': True})
    return int(a.require_no_regression and regression)


if __name__ == '__main__':
    raise SystemExit(main())
