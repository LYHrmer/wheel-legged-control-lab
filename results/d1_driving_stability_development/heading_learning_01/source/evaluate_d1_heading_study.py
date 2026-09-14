"""Evaluate all declared heading-task checkpoints on the already opened dev road.

Run in a separate process from training. Each model predicts from its own live
85-dimensional observation; every early termination is retained without reset.
This is development evaluation, not a held-out test or a complete G1 gate.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy
from scripts.run_d1_heading_study import CHECKPOINT_STEPS, TRAIN_SEEDS, source_hashes
from scripts.run_d1_wheel_common_mean_study import (
    ROOT,
    case_command,
    sha256,
    write_json,
    write_manifest,
)
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig

SCHEMA = 'd1-heading-task-development-evaluation-v1'
DEVELOPMENT_PROTOCOL = ROOT/'results/d1_wheel_common_mean_development/evaluation_protocol.json'


def run_episode(output, case, *, checkpoint=None):
    output.mkdir()
    if case['command']['target_yaw_rps'] != 0.:
        raise ValueError('this cross-track metric requires a straight reference')
    env = D1HeadingTrackingEnv(
        terrain=D1LocomotionTerrainConfig(**case['terrain']),
        episode_seconds=case['episode_seconds'], command_source=case_command(case['command']),
    )
    rows, positions, velocities, observations, applied, requested = [], [], [], [], [], []
    try:
        obs, initial = env.reset(seed=case['seed'])
        write_json(output/'episode_metadata.json', initial)
        model = None if checkpoint is None else load_heading_policy(
            checkpoint, checkpoint.with_suffix('.metadata.json'), env,
        )
        initial_y = float(env.plant.data.qpos[1])
        positions.append(env.plant.data.qpos.copy())
        velocities.append(env.plant.data.qvel.copy())
        observations.append(obs.copy())
        with gzip.open(output/'trace.jsonl.gz', 'xt') as stream:
            for _ in range(env.max_steps):
                action = np.zeros(8, dtype=np.float32) if model is None else model.predict(
                    obs, deterministic=True,
                )[0]
                obs, reward, terminated, truncated, info = env.step(action)
                row = {
                    'heading_task': info['heading_task'], 'reward': reward,
                    'reward_terms': info['reward_terms'],
                    'servo_reward': info['servo_reward'],
                    'servo_reward_terms': info['servo_reward_terms'],
                    'metrics': info['metrics'], 'terrain_exposure': info['terrain_exposure'],
                    'cross_track_after_m': float(env.last_transition.truth.base_position[1])-initial_y,
                    'terminated': bool(terminated), 'truncated': bool(truncated),
                    'terminal_reason': info['terminal_reason'],
                }
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                rows.append(row)
                positions.append(env.plant.data.qpos.copy())
                velocities.append(env.plant.data.qvel.copy())
                observations.append(obs.copy())
                applied.append(info['applied_action'].copy())
                requested.append(np.asarray(action).copy())
                if terminated or truncated:
                    break
        np.savez_compressed(
            output/'states.npz', qpos=np.asarray(positions), qvel=np.asarray(velocities),
            observations=np.asarray(observations), applied_actions=np.asarray(applied),
            requested_actions=np.asarray(requested),
        )

        def rmse(values):
            return math.sqrt(float(np.mean(np.square(values))))

        heading = [row['heading_task']['heading_error_after'] for row in rows]
        summary = {
            'checkpoint': None if checkpoint is None else str(checkpoint),
            'actual_transitions': len(rows), 'time_s': rows[-1]['metrics']['time_s'],
            'completed_duration': rows[-1]['truncated'] and not rows[-1]['terminated'],
            'terminal_reason': rows[-1]['terminal_reason'],
            'velocity_rmse_mps': rmse([row['metrics']['velocity_error_mps'] for row in rows]),
            'height_rmse_m': rmse([row['metrics']['height_error_m'] for row in rows]),
            'heading_rmse_rad': rmse(heading), 'heading_peak_rad': max(abs(v) for v in heading),
            'cross_track_peak_m': max(abs(row['cross_track_after_m']) for row in rows),
            'cross_track_final_m': rows[-1]['cross_track_after_m'],
            'user_yaw_rate_rmse_rps': rmse([
                row['heading_task']['user_yaw_rate_error_after'] for row in rows
            ]),
            'task_return': sum(row['reward'] for row in rows),
            'servo_command_return': sum(row['servo_reward'] for row in rows),
            'heading_goal_return': sum(row['reward_terms']['heading_goal'] for row in rows),
            'nonwheel_contact_steps': sum(row['metrics']['undesired_ground_contacts'] > 0 for row in rows),
            'max_abs_ground_relative_roll_pitch_rad': float(np.max(np.abs([
                [row['metrics']['roll_error_rad'], row['metrics']['pitch_error_rad']] for row in rows
            ]))),
            'terminal_state': {
                'actual_roll_pitch_rad': env.last_transition.truth.base_rpy[:2].tolist(),
                'clearance_m': rows[-1]['metrics']['clearance_m'],
                'nonwheel_contacts': rows[-1]['metrics']['undesired_ground_contacts'],
                'x_boundary': bool(abs(positions[-1][0]) >= env.episode_metadata['safe_half_size_m'][0]),
                'y_boundary': bool(abs(positions[-1][1]) >= env.episode_metadata['safe_half_size_m'][1]),
            },
            'final_qpos': positions[-1].tolist(),
            'return_comparison_limit': 'new task reward differs from legacy; use common physical metrics',
            'classification': 'opened development case; no held-out or full G1 claim',
        }
        write_json(output/'summary.json', summary)
        write_manifest(output)
        return summary
    finally:
        env.close()


def common_prefix(output, labels):
    """Keep shorter failures from gaining an artificial full-duration RMSE advantage."""
    traces = {}
    for label in labels:
        with gzip.open(output/label/'trace.jsonl.gz', 'rt') as stream:
            traces[label] = [json.loads(line) for line in stream]
    pairs = []
    if 'zero' in traces:
        pairs.extend(('zero', label) for label in labels if label != 'zero')
    for seed in TRAIN_SEEDS:
        early, late = f'seed{seed}_step16384', f'seed{seed}_step65536'
        if early in traces and late in traces:
            pairs.append((early, late))
    comparisons = []
    for first, second in pairs:
        count = min(len(traces[first]), len(traces[second]))
        metrics = {}
        for label in (first, second):
            rows = traces[label][:count]
            values = {
                'velocity_rmse_mps': [r['metrics']['velocity_error_mps'] for r in rows],
                'height_rmse_m': [r['metrics']['height_error_m'] for r in rows],
                'heading_rmse_rad': [r['heading_task']['heading_error_after'] for r in rows],
                'cross_track_rmse_m': [r['cross_track_after_m'] for r in rows],
            }
            metrics[label] = {key: math.sqrt(float(np.mean(np.square(value))))
                              for key, value in values.items()}
            metrics[label]['cross_track_final_m'] = rows[-1]['cross_track_after_m']
        comparisons.append({'first': first, 'second': second,
                            'actual_common_transitions': count,
                            'time_s': traces[first][count-1]['metrics']['time_s'],
                            'metrics': metrics})
    return {'definition': 'Both branches use only their shared actual observed prefix; no padding.',
            'comparisons': comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(TRAIN_SEEDS))
    parser.add_argument('--steps', type=int, nargs='+', default=list(CHECKPOINT_STEPS))
    parser.add_argument('--skip-zero', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(TRAIN_SEEDS):
        parser.error('seeds must be a unique subset of the declared training seeds')
    if len(set(args.steps)) != len(args.steps) or not set(args.steps) <= set(CHECKPOINT_STEPS):
        parser.error('steps must be a unique subset of the declared checkpoint budgets')
    case = next(c for c in json.loads(DEVELOPMENT_PROTOCOL.read_text())['cases']
                if c['name'] == 'dev_straight_road' and c['split'] == 'dev')
    inputs = {str(ROOT/name): digest for name, digest in source_hashes().items()}
    for item in (Path(__file__).resolve(), DEVELOPMENT_PROTOCOL, args.training_root/'protocol.json'):
        inputs[str(item.resolve())] = sha256(item)
    checkpoints = []
    for seed in args.seeds:
        for step in args.steps:
            path = args.training_root/f'seed{seed}/checkpoints/step{step}/model.zip'
            for item in (path, path.with_suffix('.metadata.json')):
                inputs[str(item.resolve())] = sha256(item)
            checkpoints.append((f'seed{seed}_step{step}', path))
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'protocol.json', {
        'schema': SCHEMA, 'case': case, 'input_sha256': inputs,
        'models': [label for label, _ in checkpoints], 'include_zero': not args.skip_zero,
        'failure_rule': 'first environment termination; no reset, missing samples never padded',
        'classification': 'development only, fixed physical metrics shared with the old heading probe',
    })
    episodes = {}
    for label, checkpoint in ([] if args.skip_zero else [('zero', None)])+checkpoints:
        episodes[label] = run_episode(args.output/label, case, checkpoint=checkpoint)
        print(json.dumps({'case': label, **episodes[label]}, allow_nan=False), flush=True)
    changed = [name for name, digest in inputs.items() if sha256(name) != digest]
    if changed:
        raise RuntimeError(f'evaluation inputs changed: {changed}')
    write_json(args.output/'common_prefix.json', common_prefix(args.output, list(episodes)))
    write_json(args.output/'summary.json', {
        'schema': SCHEMA, 'input_unchanged': True, 'episodes': episodes,
    })
    write_manifest(args.output)


if __name__ == '__main__':
    main()
