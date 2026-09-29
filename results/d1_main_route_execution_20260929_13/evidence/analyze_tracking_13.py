"""Read closed 11-S data once: no engine, policy, Torch, or experiment imports."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent.parent
RUN = W / 'rl11/training_run_01'
READER = W / 'rl11/partial_readback_20260929_05.json'
MANIFEST = W / 'rl11/partial_closed_manifest_20260929_05.json'
TERMS = ('tracking_vx', 'tracking_vy', 'tracking_yaw', 'height', 'attitude',
         'actual_torque', 'action_change', 'leg_speed', 'termination')
MASKS = ('position_clip_mask', 'rate_limit_mask', 'wheel_speed_target_clip_mask',
         'torque_clip_mask', 'protection_changed_mask')


def identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    assert a.size and np.isfinite(a).all()
    return {'n': a.size, 'mean': float(a.mean()), 'std': float(a.std()),
            'min': float(a.min()), 'max': float(a.max()),
            'p05': float(np.quantile(a, .05)), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95))}


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze(output: Path) -> dict:
    assert not output.exists()
    manifest = json.loads(MANIFEST.read_text())['files']
    verified = {}

    def checked(rel):
        path = RUN / rel
        actual = identity(path)
        assert actual == manifest[rel], ('closed input changed', rel)
        verified[rel] = actual
        return path

    prior = json.loads(READER.read_text())
    assert prior['complete_case_count'] == 9 and prior['bitwise_initial_pairs_verified'] == 4
    scores = prior['individual_complete_case_scores_not_matched_pair_aggregate'][:8]
    reports = []
    for score in scores:
        assert score['task_passed'] is True
        folder = 'heldout/' + score['case_id'] + '_' + score['actor']
        receipt = json.loads(checked(folder + '/case_receipt.json').read_text())
        records = []
        for block in receipt['controller_record_blocks']:
            with gzip.open(checked(folder + '/' + block['file']), 'rt') as stream:
                records.extend(json.loads(line) for line in stream)
        assert len(records) == 1600 and [r['tick'] for r in records] == list(range(1600))
        info = [r['info'] for r in records]
        calc = [i['controller_record']['calculation'] for i in info]
        vx = np.array([i['metrics']['body_com_vx_mps'] for i in info])
        yaw = np.array([i['metrics']['body_yaw_rate_rps'] for i in info])
        servo = np.array([i['consumed_command']['forward_velocity_mps'] for i in info])
        servo_yaw = np.array([i['consumed_command']['yaw_rate_rps'] for i in info])
        err, yawerr = vx-servo, yaw-servo_yaw
        wheel = np.array([c['wheel_action_offset_rad_s'] for c in calc])
        leg = np.array([c['leg_action_offset_rad'] for c in calc])
        nominal = np.array([c['consumed_nominal_wheel_speed_rad_s'] for c in calc])
        reward_terms = np.array([[i['reward_terms'][k] for k in TERMS] for i in info])
        assert np.allclose(reward_terms.sum(axis=1), [r['reward'] for r in records],
                           atol=1e-10, rtol=0)
        hold_begin, hold_end = score['windows']['hold']
        release = score['windows']['release_tick_t0']
        assert hold_end == release
        windows = {'acceleration': (175, hold_begin), 'hold': (hold_begin, hold_end),
                   'drive': (175, release), 'post_release': (release, 1600)}
        row = {'case_id': score['case_id'], 'actor': score['actor'], 'windows': {}}
        for label, (begin, end) in windows.items():
            sl = slice(begin, end)
            ev, ey = err[sl], yawerr[sl]
            native_speed_sse = float(np.sum((ev/.25)**2))
            yaw_sse = float(np.sum((ey/.4)**2))
            bias_sse = len(ev) * float(ev.mean()**2)/.25**2
            data = {'ticks': [begin, end], 'velocity_error_mps': stats(ev),
                    'rms_velocity_error_mps': float(np.sqrt(np.mean(ev**2))),
                    'normalized_sse_vx': native_speed_sse, 'normalized_sse_yaw': yaw_sse,
                    'normalized_sse_total': native_speed_sse+yaw_sse,
                    'vx_bias_fraction_of_sse': bias_sse/native_speed_sse
                    if native_speed_sse else 0.0,
                    'wheel_offset_mean_rad_s': wheel[sl].mean(axis=0).tolist(),
                    'common_wheel_offset_rad_s': stats(wheel[sl].mean(axis=1)),
                    'leg_offset_mean_rad': leg[sl].mean(axis=0).tolist(),
                    'leg_offset_abs_mean_rad': float(np.abs(leg[sl]).mean()),
                    'nominal_wheel_mean_rad_s': nominal[sl].mean(axis=0).tolist(),
                    'reward_terms_mean': dict(zip(TERMS, reward_terms[sl].mean(axis=0).tolist())),
                    'reward_mean': float(reward_terms[sl].sum(axis=1).mean()),
                    'mask_component_fraction': {
                        k: float(np.asarray([c[k] for c in calc[sl]], dtype=bool).mean())
                        for k in MASKS},
                    'action_gate_enabled_fraction': float(np.mean([
                        i['controller_record']['action_gate_enabled'] for i in info[sl]])),
                    'common_wheel_vs_vx_error_correlation_not_causal': correlation(
                        wheel[sl].mean(axis=1), ev)}
            row['windows'][label] = data
        assert np.isclose(row['windows']['drive']['normalized_sse_total'],
                          score['rl_terms']['sse_total'], rtol=0, atol=1e-9)
        reports.append(row)
    pairs = []
    for i in range(4):
        zero, policy = reports[2*i:2*i+2]
        assert zero['case_id'] == policy['case_id']
        pairs.append({'case_id': zero['case_id'], 'windows': {
            phase: {'policy_minus_zero_vx_bias_mps':
                    policy['windows'][phase]['velocity_error_mps']['mean']
                    - zero['windows'][phase]['velocity_error_mps']['mean'],
                    'policy_minus_zero_reward': policy['windows'][phase]['reward_mean']
                    - zero['windows'][phase]['reward_mean'],
                    'policy_minus_zero_reward_terms': {
                        key: policy['windows'][phase]['reward_terms_mean'][key]
                        - zero['windows'][phase]['reward_terms_mean'][key] for key in TERMS},
                    'sse_ratio': policy['windows'][phase]['normalized_sse_total']
                    / zero['windows'][phase]['normalized_sse_total']}
            for phase in zero['windows']}})

    train_manifest = json.loads(checked('training/training_blocks_manifest.json').read_text())
    selected = ('reward', 'reward_terms9', 'episode_index', 'episode_tick',
                'terminated', 'truncated', 'effective_action16')
    pieces = {key: [] for key in selected}
    for block in train_manifest['numeric_blocks']:
        path = checked('training/training_numeric_blocks/' + block['file'])
        with np.load(path, allow_pickle=False) as arrays:
            for key in selected:
                pieces[key].append(arrays[key])
    train = {key: np.concatenate(values) for key, values in pieces.items()}
    assert len(train['reward']) == 65536
    assert np.allclose(train['reward_terms9'].sum(axis=1), train['reward'], atol=1e-10)
    with zipfile.ZipFile(checked('final_checkpoint/final_model.zip')) as archive:
        saved_config = json.loads(archive.read('data'))
    hyper = {k: saved_config[k] for k in ('gamma', 'gae_lambda', 'vf_coef', 'max_grad_norm',
                                       'ent_coef', 'target_kl', 'learning_rate',
                                       'normalize_advantage')}
    proxy = []
    for episode in range(65):
        indices = np.flatnonzero(train['episode_index'] == episode)
        assert len(indices) == 1000 and train['truncated'][indices[-1]]
        ret = 0.0
        for index in indices[::-1]:
            ret = float(train['reward'][index]) + hyper['gamma'] * ret
        proxy.append(ret)
    learning = json.loads(checked('training/learning_receipt.json').read_text())
    updates = learning['updates']
    training = {'controls': 65536, 'reward_per_tick': stats(train['reward']),
                'reward_terms_mean': dict(zip(TERMS, train['reward_terms9'].mean(axis=0).tolist())),
                'terminated_count': int(train['terminated'].sum()),
                'truncated_count': int(train['truncated'].sum()),
                'discounted_zero_tail_episode_reward_sum': stats(proxy),
                'return_boundary': 'zero-tail descriptive sums; NOT PPO GAE/bootstrap targets',
                'actual_saved_config_json_no_unpickle': hyper,
                'optimizer_postclip_grad_norm_means': stats([
                    u['grad_norm_at_step_mean'] for u in updates]),
                'value_loss': stats([u['train/value_loss'] for u in updates]),
                'explained_variance': stats([u['train/explained_variance'] for u in updates]),
                'approx_kl': stats([u['train/approx_kl'] for u in updates]),
                'missing': ['historical preclip actor/value gradient norms',
                            'historical minibatch values, bootstrap values, GAE targets']}
    result = {'schema': 'd1-saved-tracking-decomposition-13-v1', 'cases': reports,
              'pairs': pairs, 'training': training, 'verified_input_files': verified,
              'readback_identity': identity(READER), 'raw_manifest_identity': identity(MANIFEST),
              'analysis_source': identity(Path(__file__)),
              'physics_controls': 0, 'policy_loads': 0, 'policy_predictions': 0,
              'optimizer_steps': 0, 'causal_counterfactual_test_performed': False}
    assert not any(name in sys.modules for name in ('torch', 'mujoco', 'stable_baselines3'))
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.output)
    print(json.dumps({'cases': len(result['cases']), 'pairs': len(result['pairs']),
                      'verified_input_files': len(result['verified_input_files']),
                      'physics_controls': 0, 'model_calls': 0}))
