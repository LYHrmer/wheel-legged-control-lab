"""One fixed-seed, read-only inspection of existing artifacts; outputs only in /tmp."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

run = Path('/home/lyh/wheel-legged-control-lab/results/d1_budget_study/shared2_seed31000')
out = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (run / 'updates/updates.jsonl').read_text().splitlines()]
peak = max(rows, key=lambda row: row['logger_train_metrics']['train/value_loss'])
index = peak['audit_index']
end = peak['num_timesteps']
begin = end - 512


def stats(array):
    array = np.asarray(array)
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'all_finite': bool(np.isfinite(array).all()),
            'min': float(array.min()), 'max': float(array.max()),
            'mean': float(array.mean()), 'variance': float(array.var())}


with np.load(run / 'updates' / peak['npz'], allow_pickle=False) as npz:
    z = {key: npz[key].copy() for key in npz.files}
rewards, values = z['gae_rewards'], z['gae_values']
starts = z['gae_episode_starts']
gamma, lam = float(z['gae_gamma']), float(z['gae_lambda'])
advantages = np.zeros_like(values)
last_gae = 0
for t in reversed(range(len(rewards))):
    if t == len(rewards) - 1:
        next_nonterminal = 1.0 - z['gae_last_dones'].astype(np.float32)
        next_values = z['gae_last_values']
    else:
        next_nonterminal = 1.0 - starts[t + 1]
        next_values = values[t + 1]
    delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
    last_gae = delta + gamma * lam * next_nonterminal * last_gae
    advantages[t] = last_gae
returns = advantages + values
flat_returns = returns.T.reshape(-1)
flat_values = values.T.reshape(-1)
full_ev = float(1 - np.var(flat_returns - flat_values) / np.var(flat_returns))
subset_ev = 1 - float(np.var(z['returns'] - z['old_values'])) / float(np.var(z['returns']))
done_locations = [(int(t - 1), int(worker)) for t, worker in np.argwhere(starts[1:] > 0) + [1, 0]]
done_locations += [(127, int(worker)) for worker in np.flatnonzero(z['gae_last_dones'])]

csv_rows = []
with (run / 'training_samples.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        sample = int(row['sample'])
        if sample > end + 4:
            break
        if begin <= sample <= end + 4:
            csv_rows.append(row)
rollout = [row for row in csv_rows if begin < int(row['sample']) <= end]
assert len(rollout) == 512
raw_rewards = np.zeros_like(rewards)
for row in rollout:
    raw_rewards[(int(row['sample']) - begin) // 4 - 1, int(row['worker'])] = float(row['reward'])
terminal_rows = [row for row in rollout if row['terminated'] == 'True']
selected = ('worker', 'sample', 'time_s', 'terminated', 'reward', 'reward_termination',
            'x_m', 'y_m', 'z_m', 'clearance_m', 'roll_error_rad', 'pitch_error_rad',
            'undesired_ground_contacts', 'velocity_mps', 'height_error_m')
events = []
for row in terminal_rows:
    worker = int(row['worker'])
    sample = int(row['sample'])
    history = [x for x in csv_rows if int(x['worker']) == worker and abs(int(x['sample']) - sample) <= 4]
    with (run / f'worker{worker}.monitor.csv').open(newline='') as stream:
        next(stream)
        monitor = list(csv.DictReader(stream))
    cum_length = 0
    episodes = [json.loads(line) for line in (run / f'worker{worker}_episodes.jsonl').read_text().splitlines()]
    match = None
    for episode, entry in enumerate(monitor):
        cum_length += int(entry['l'])
        if cum_length * 4 == sample:
            match = {'episode': episode, 'monitor_return': float(entry['r']),
                     'monitor_length_steps': int(entry['l']), 'cumulative_worker_steps': cum_length,
                     'episode_start_metadata': {key: episodes[episode][key] for key in
                        ('episode', 'command_seed', 'measurement_seed', 'duration_s', 'terrain')},
                     'next_reset_metadata': {key: episodes[episode + 1][key] for key in
                        ('episode', 'command_seed', 'measurement_seed', 'duration_s', 'terrain')}}
            break
    events.append({'worker': worker, 'sample': sample,
                   'rollout_time_index_zero_based': (sample - begin) // 4 - 1,
                   'neighbors': [{key: x[key] for key in selected} for x in history],
                   'monitor_episode_match': match})

worker_stats = []
for worker in range(4):
    worker_stats.append({'worker': worker, 'rewards': stats(rewards[:, worker]),
        'old_values': stats(values[:, worker]), 'returns': stats(returns[:, worker]),
        'pre_update_mse': float(np.mean((returns[:, worker] - values[:, worker]) ** 2)),
        'episode_start_time_indices': np.flatnonzero(starts[:, worker]).tolist(),
        'last_done': bool(z['gae_last_dones'][worker])})
input_files = [run / 'updates/updates.jsonl', run / 'updates' / peak['npz'],
               run / 'training_samples.csv', run / 'rollout_exposure.json']
for event in events:
    worker = event['worker']
    input_files.extend([run / f'worker{worker}.monitor.csv', run / f'worker{worker}_episodes.jsonl'])
result = {'selection': 'Preselected shared2_seed31000; maximum train/value_loss within its 512 updates; no holdout selection.',
          'run': str(run), 'update': peak, 'actual_environment_transition_interval': [begin + 1, end],
          'csv_sample_label_range': [begin + 4, end],
          'vector_step_range_per_worker_one_based': [begin // 4 + 1, end // 4],
          'full_rollout_samples': 512, 'diagnostic_subset_samples': 128,
          'subset_identity': 'worker0, all128 steps; first128 in env-major order',
          'full_returns_status': 'Reconstructed by NumPy GAE from stored full arrays; NPZ returns itself is subset only.',
          'all_npz_arrays_finite': all(np.isfinite(x).all() for x in z.values()),
          'full_stats': {'gae_rewards': stats(rewards), 'old_values': stats(values),
                         'reconstructed_returns': stats(returns), 'reconstructed_advantages': stats(advantages)},
          'full_ev_recomputed': full_ev, 'full_ev_logger': peak['logger_train_metrics']['train/explained_variance'],
          'full_ev_abs_error': abs(full_ev - peak['logger_train_metrics']['train/explained_variance']),
          'subset_ev_recomputed': subset_ev, 'subset_ev_jsonl': peak['explained_variance_old_values'],
          'subset_returns_max_abs_error': float(np.max(np.abs(flat_returns[:128] - z['returns']))),
          'subset_advantages_max_abs_error': float(np.max(np.abs(advantages.T.reshape(-1)[:128] - z['advantages']))),
          'raw_vs_gae_reward_max_abs_error_float32': float(np.max(np.abs(raw_rewards - rewards))),
          'pre_update_full_mse': float(np.mean((returns - values) ** 2)),
          'pre_update_error_sum_by_worker': np.sum((returns - values) ** 2, axis=0).tolist(),
          'worker0_preterminal_error_share': float(np.sum((returns[:40, 0] - values[:40, 0]) ** 2)
                                                 / np.sum((returns - values) ** 2)),
          'terminal_gae_values': {'old_value': float(values[39, 0]), 'return': float(returns[39, 0]),
                                  'advantage': float(advantages[39, 0]), 'reward': float(rewards[39, 0])},
          'done_locations_inferred_from_next_starts_and_last_dones': done_locations,
          'episode_start_locations': np.argwhere(starts > 0).tolist(), 'gae_last_dones': z['gae_last_dones'].tolist(),
          'csv_terminated_count': len(terminal_rows), 'terminal_events': events, 'worker_stats': worker_stats,
          'rollout_exposure': json.loads((run / 'rollout_exposure.json').read_text())[index],
          'neighbor_update_metrics': [{k: r[k] for k in ('audit_index', 'num_timesteps', 'logger_train_metrics')}
                                      for r in rows[max(0,index-1):index+2]],
          'input_sha256': {}}
for path in input_files:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    result['input_sha256'][str(path)] = digest.hexdigest()
(out / 'peak_diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: result[key] for key in ('update', 'full_stats', 'full_ev_recomputed',
    'subset_ev_recomputed', 'pre_update_full_mse', 'done_locations_inferred_from_next_starts_and_last_dones',
    'terminal_events', 'worker_stats')}, indent=2))
