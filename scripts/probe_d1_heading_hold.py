"""Development-only closed-loop heading intervention on saved D1 policies.

This changes commands seen by the frozen policy, not its weights. The existing
82-value observation and servo-command reward stay intact. User heading/rate
errors are measured separately, using the reference for the executed interval.
Only the previously opened straight development road is supported here.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from scripts.d1_heading_reference import HeadingReference, heading_feedback, wrap_angle
from scripts.run_d1_wheel_common_mean_study import (
    ROOT,
    case_command,
    sha256,
    source_hashes,
    write_json,
    write_manifest,
)
from wheel_legged_control.d1.locomotion_checkpoint import load_locomotion_policy
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig

SCHEMA = 'd1-heading-outer-loop-development-v1'
STUDY = ROOT / 'results/d1_wheel_common_mean_development'


def run_episode(output, case, *, mode, checkpoint=None):
    """Each mode owns a new environment and predicts from its own current obs."""
    output.mkdir()
    user_source = case_command(case['command'])
    reference = HeadingReference()
    reference.reset(0.0)
    latched = {}
    env = None

    def source(time_s):
        user = user_source(time_s)
        ref = reference.advance(user.yaw_rate_rps, time_s)
        state = env.loop.provider.read()
        feedback = heading_feedback(
            ref.heading_rad, float(state.base_rpy[2]),
            float(state.base_angular_velocity_body[2]), user.yaw_rate_rps,
        )
        servo = replace(user, yaw_rate_rps=feedback.servo_yaw_rate_rps) if mode == 'hold' else user
        latched.clear()
        latched.update(user=asdict(user), reference=asdict(ref), servo=asdict(servo))
        return servo

    env = D1LocomotionEnv(
        baseline='wheel_leg', action_mode='independent8',
        terrain=D1LocomotionTerrainConfig(**case['terrain']),
        command_source=source, episode_seconds=case['episode_seconds'],
    )
    rows, positions, velocities, observations, actions = [], [], [], [], []
    try:
        obs, initial = env.reset(seed=case['seed'])
        write_json(output/'episode_metadata.json', initial)
        model = None if checkpoint is None else load_locomotion_policy(
            checkpoint, checkpoint.with_suffix('.metadata.json'), env,
        )
        initial_y = float(env.plant.data.qpos[1])
        positions.append(env.plant.data.qpos.copy())
        velocities.append(env.plant.data.qvel.copy())
        observations.append(obs.copy())
        with (output/'trace.jsonl').open('x') as stream:
            for _ in range(env.max_steps):
                before = dict(latched)
                action = np.zeros(8, dtype=np.float32) if model is None else model.predict(
                    obs, deterministic=True,
                )[0]
                obs, reward, terminated, truncated, info = env.step(action)
                truth = env.last_transition.truth
                dt = env.plant.control_dt
                target_after = wrap_angle(
                    before['reference']['heading_rad']+dt*before['user']['yaw_rate_rps']
                )
                row = {
                    'user_command_before': before['user'],
                    'servo_command_before': before['servo'],
                    'reference_before': before['reference'],
                    'reference_heading_after_rad': target_after,
                    'heading_error_after_rad': wrap_angle(float(truth.base_rpy[2])-target_after),
                    'cross_track_after_m': float(truth.base_position[1])-initial_y,
                    'user_yaw_rate_error_rps': float(truth.base_angular_velocity_body[2])
                    -before['user']['yaw_rate_rps'],
                    'servo_reward': reward,
                    'reward_terms': info['reward_terms'],
                    'metrics': info['metrics'],
                    'terminal_reason': info['terminal_reason'],
                    'terminated': bool(terminated), 'truncated': bool(truncated),
                }
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                rows.append(row)
                positions.append(env.plant.data.qpos.copy())
                velocities.append(env.plant.data.qvel.copy())
                observations.append(obs.copy())
                actions.append(info['applied_action'].copy())
                if terminated or truncated:
                    break
        np.savez_compressed(
            output/'states.npz', qpos=np.asarray(positions), qvel=np.asarray(velocities),
            observations=np.asarray(observations), applied_actions=np.asarray(actions),
        )
        def rmse(values):
            return math.sqrt(float(np.mean(np.square(values))))
        summary = {
            'mode': mode, 'checkpoint': None if checkpoint is None else str(checkpoint),
            'actual_transitions': len(rows), 'time_s': rows[-1]['metrics']['time_s'],
            'terminal_reason': rows[-1]['terminal_reason'],
            'completed_duration': rows[-1]['truncated'] and not rows[-1]['terminated'],
            'velocity_rmse_mps': rmse([r['metrics']['velocity_error_mps'] for r in rows]),
            'height_rmse_m': rmse([r['metrics']['height_error_m'] for r in rows]),
            'heading_rmse_rad': rmse([r['heading_error_after_rad'] for r in rows]),
            'heading_peak_rad': max(abs(r['heading_error_after_rad']) for r in rows),
            'cross_track_peak_m': max(abs(r['cross_track_after_m']) for r in rows),
            'cross_track_final_m': rows[-1]['cross_track_after_m'],
            'user_yaw_rate_rmse_rps': rmse([r['user_yaw_rate_error_rps'] for r in rows]),
            'servo_command_return': sum(r['servo_reward'] for r in rows),
            'servo_return_comparison_limit': 'targets differ across modes; not a user-goal score',
        }
        write_json(output/'summary.json', summary)
        write_manifest(output)
        return summary
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[48001, 48002, 48003])
    parser.add_argument('--skip-zero', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= {48001, 48002, 48003}:
        parser.error('seeds must be a unique subset of 48001, 48002, 48003')
    import torch
    torch.set_num_threads(1)
    protocol_path = STUDY/'evaluation_protocol.json'
    case = next(c for c in json.loads(protocol_path.read_text())['cases']
                if c['name'] == 'dev_straight_road' and c['split'] == 'dev')
    if case['command']['target_yaw_rps'] != 0:
        raise ValueError('cross-track metric requires the declared straight reference')
    inputs = source_hashes()
    for path in (Path(__file__).resolve(), ROOT/'scripts/d1_heading_reference.py', protocol_path):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    checkpoints = []
    for seed in args.seeds:
        for variant in ('unbounded', 'bounded'):
            path = STUDY/f'models/seed{seed}/{variant}/model.zip'
            checkpoints.append((f'seed{seed}_{variant}', path))
            for item in (path, path.with_suffix('.metadata.json')):
                inputs[str(item.relative_to(ROOT))] = sha256(item)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'protocol.json', {
        'schema': SCHEMA, 'case': case, 'input_sha256': inputs,
        'models': [label for label, _ in checkpoints], 'include_zero': not args.skip_zero,
        'modes': ['original', 'hold'], 'heading_kp': 2., 'heading_kd': .4,
        'heading_rate_limit_rps': 1.,
        'classification': 'command intervention on frozen policies; no training or held-out evaluation',
        'reference': 'simulation-time left hold; fixed spawn heading 0; straight reference through spawn',
        'failure_rule': 'first environment termination stops each branch; no reset or padding',
    })
    summaries = {}
    work = ([] if args.skip_zero else [('zero', None)])+checkpoints
    for label, checkpoint in work:
        for mode in ('original', 'hold'):
            key = f'{label}_{mode}'
            summaries[key] = run_episode(args.output/key, case, mode=mode, checkpoint=checkpoint)
            print(json.dumps({'case': key, **summaries[key]}, allow_nan=False), flush=True)
    changed = [name for name, digest in inputs.items() if sha256(ROOT/name) != digest]
    if changed:
        raise RuntimeError(f'inputs changed during the study: {changed}')
    write_json(args.output/'summary.json', {'schema': SCHEMA, 'input_unchanged': True,
                                          'episodes': summaries})
    write_manifest(args.output)


if __name__ == '__main__':
    main()
