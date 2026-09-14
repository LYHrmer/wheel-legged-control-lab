"""Independently recompute final-policy summaries from preserved transitions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.input
    manifest = json.loads((source/'manifest.json').read_text())
    assert all((source/name).stat().st_size == entry['bytes']
               and sha(source/name) == entry['sha256'] for name, entry in manifest.items())
    protocol = json.loads((source/'protocol.json').read_text())
    assert all(sha(args.repo/name) == value for name, value in protocol['source_sha256'].items())
    assert all(sha(Path(name)) == value for name, value in protocol['model_sha256'].items())
    initials, commands, summaries = [], [], []
    for name in protocol['policies_in_run_order']:
        case = source/name
        initial = json.loads((case/'initial.json').read_text())
        summary = json.loads((case/'summary.json').read_text())
        with (case/'trace.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        initials.append(initial)
        commands.append([r['command'] for r in rows])
        assert len(rows) == summary['actual_transitions'] == 3200
        assert not any(r['terminated'] or r['truncated'] for r in rows[:-1])
        assert rows[-1]['truncated'] and not rows[-1]['terminated']
        assert rows[-1]['terminal_reason'] == 'time_limit'
        for index, row in enumerate(rows):
            assert row['transition'] == index+1
            assert abs(math.fsum(row['reward_terms'].values())-row['reward']) < 1e-12
            assert np.array_equal(row['actor_action'], row['policy_action'])
            expected_forward = .25*max(0., min(1., (index*.01-.5)/.5))
            assert abs(row['command']['forward_velocity_mps']-expected_forward) < 1e-12
            assert row['command']['yaw_rate_rps'] == 0
            assert row['command']['clearance_m'] == .455
        recomputed = {}
        for field, key in [('velocity_error_mps', 'velocity_rmse_mps'),
                           ('yaw_rate_error_rps', 'yaw_rate_rmse_rps'),
                           ('height_error_m', 'height_rmse_m')]:
            value = math.sqrt(math.fsum(r['metrics'][field]**2 for r in rows)/len(rows))
            assert abs(value-summary[key]) < 1e-12
            recomputed[key] = value
        assert abs(math.fsum(r['reward'] for r in rows)-summary['cumulative_return']) < 1e-10
        for term, expected in summary['cumulative_reward_terms'].items():
            assert abs(math.fsum(r['reward_terms'][term] for r in rows)-expected) < 1e-10
        nonflat = sum(int(r['terrain_exposure']['nonflat_now']) for r in rows)
        assert nonflat == summary['geometric_nonflat_exposure']['nonflat_steps']
        with np.load(case/'states.npz', allow_pickle=False) as states:
            assert states['qpos'].shape[0] == len(rows)+1
            assert states['observation'].shape == (len(rows)+1, 82)
            assert all(np.isfinite(states[key]).all() for key in states.files)
            assert np.allclose(np.diff(states['time_s']), .01, rtol=0, atol=2e-12)
            assert np.allclose(states['qpos'][1:, :3],
                               [[r['metrics'][key] for key in ('x_m', 'y_m', 'z_m')] for r in rows],
                               rtol=0, atol=1e-12)
            progress = float(states['qpos'][-1, 0]-states['qpos'][0, 0])
            assert abs(progress-summary['signed_forward_progress_m']) < 1e-12
        summaries.append({'policy': name, 'transitions': len(rows), 'nonflat_steps': nonflat,
                          'signed_progress_from_integrated_qpos_m': progress,
                          'metrics_recomputed': recomputed, 'summary_matches_trace_and_state': True})
    assert all(value == initials[0] for value in initials)
    assert all(value == commands[0] for value in commands)
    report = {'status': 'passed', 'policies': summaries,
              'manifest_files_verified': len(manifest),
              'source_files_verified': len(protocol['source_sha256']),
              'checkpoint_and_metadata_files_verified': len(protocol['model_sha256']),
              'initial_observation_state_metadata_identical': True,
              'all_3200_commands_identical_across_policies': True,
              'physics_steps_performed_by_audit': 0,
              'primary_manifest_sha256': sha(source/'manifest.json'),
              'audit_source_sha256': sha(Path(__file__))}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
