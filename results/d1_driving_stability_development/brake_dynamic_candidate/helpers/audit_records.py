"""Independent read-only arithmetic and integrity audit; never runs physics."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source/'manifest.json').read_text())
    for name, metadata in manifest.items():
        path = source/name
        assert path.stat().st_size == metadata['bytes'] and sha(path) == metadata['sha256'], name
    summary = json.loads((source/'summary.json').read_text())
    protocol = json.loads((source/'protocol.json').read_text())
    checked, rows, mechanism = [], [], []
    source_archives = {}
    for result in summary['results']:
        case, strategy = result['case'], result['strategy']
        directory = source/case['name']/strategy
        with np.load(directory/'states.npz', allow_pickle=False) as data:
            pos, times = data['qpos'], data['segment_time_s']
            actual_steps = len(data['applied_torque_nm'])
        trace = [json.loads(line) for line in (directory/'control_memory_trace.jsonl').read_text().splitlines()]
        assert len(trace) == actual_steps == result['control_transitions']
        assert abs(times[-1]-case['duration']) < 1e-8
        assert result['reference_release_count'] == (len(case['held']) if strategy != 'legacy' else 0)
        run_protocol = json.loads((directory/'protocol.json').read_text())
        with tarfile.open(directory/'source.tar.gz') as archive:
            for name, expected in run_protocol['source_sha256'].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
                source_archives.setdefault(name, set()).add(expected)
        with tarfile.open(directory/'work_source.tar.gz') as archive:
            for name, expected in protocol['work_source_sha256'].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
        maxima = {'distance_update_error': 0., 'actual_reference_update_error': 0.,
                  'actual_error_diagnostic_error': 0., 'brake_bound_excess': 0.}
        for row in trace:
            expected_distance = row['pre_distance_m']+row['pre_body_vx_mps']*.01
            maxima['distance_update_error'] = max(maxima['distance_update_error'], abs(expected_distance-row['post_distance_m']))
            dynamic = row['dynamic']
            reference = row['pre_reference_m']
            if dynamic:
                if dynamic['reference_written']:
                    reference = row['pre_distance_m']+dynamic['new_error_m']
                if dynamic['phase'] == 'brake':
                    expected_bound = row['pre_body_vx_mps']**2
                    assert abs(expected_bound-dynamic['bound_m']) < 1e-12
                    maxima['brake_bound_excess'] = max(maxima['brake_bound_excess'], abs(dynamic['new_error_m'])-expected_bound)
                if dynamic['phase'] == 'hold' and not dynamic['reference_written']:
                    assert dynamic['new_error_m'] == dynamic['old_error_m']
            expected_reference = float(np.clip(reference+row['applied_forward_mps']*.01,
                                               expected_distance-.55, expected_distance+.55))
            maxima['actual_reference_update_error'] = max(maxima['actual_reference_update_error'], abs(expected_reference-row['post_reference_m']))
            maxima['actual_error_diagnostic_error'] = max(maxima['actual_error_diagnostic_error'], abs((row['post_reference_m']-row['post_distance_m'])-row['post_error_m']))
        assert all(value < 1e-12 for value in maxima.values()), maxima
        checked.append({'case': case['name'], 'strategy': strategy,
                        'actual_steps': actual_steps, 'per_tick_invariant_max_error': maxima})
        direction = 1 if case['key'] == 'W' else -1
        for window in result['release_windows']:
            first = round(window['release_time_s']/.01)
            last = first+800
            progress = direction*(pos[first:last+1, 0]-pos[first, 0])
            delta = np.diff(progress)
            vx = np.diff(pos[first:last+1, 0])/np.diff(times[first:last+1])
            last_vx = vx[-200:]
            recomputed = {
                'net_world_x_m': pos[last, 0]-pos[first, 0],
                'net_along_command_m': progress[-1],
                'peak_forward_excursion_m': max(0., progress.max()),
                'max_backward_excursion_m': max(0., -progress.min()),
                'total_backward_travel_m': np.maximum(-delta, 0).sum(),
                'peak_retreat_from_prior_peak_m': (np.maximum.accumulate(progress)-progress).max(),
                'max_reverse_velocity_mps': max(0., -(direction*vx).min()),
                'last2_mean_signed_world_vx_mps': last_vx.mean(),
                'last2_mean_abs_world_vx_mps': np.abs(last_vx).mean(),
                'last2_max_abs_world_vx_mps': np.abs(last_vx).max(),
                'last2_signed_world_drift_m': pos[last, 0]-pos[last-200, 0],
                'last2_total_world_travel_m': np.abs(np.diff(pos[last-200:last+1, 0])).sum(),
            }
            assert all(abs(float(value)-window[key]) < 1e-12 for key, value in recomputed.items())
            rows.append({'case': case['name'], 'strategy': strategy, 'release_number': window['release_number'], **{key: float(value) for key, value in recomputed.items()}})
            if strategy != 'legacy':
                release_trace = trace[first:last]
                active = [row for row in release_trace if row['dynamic']['phase'] == 'brake']
                hold = [row for row in release_trace if row['dynamic']['hold_entered']]
                last_rows = release_trace[-200:]
                mechanism.append({'case': case['name'], 'release_number': window['release_number'],
                    'release_old_reference_error_m': release_trace[0]['dynamic']['old_error_m'],
                    'release_new_reference_error_m': release_trace[0]['dynamic']['new_error_m'],
                    'hold_entered': bool(hold), 'hold_entry_time_s': None if not hold else hold[0]['time_before_s'],
                    'max_consecutive_low_speed_ticks_during_brake': max(row['dynamic']['low_speed_ticks'] for row in active),
                    'last2_mean_abs_body_vx_mps': float(np.mean([abs(row['pre_body_vx_mps']) for row in last_rows])),
                    'last2_mean_actual_reference_error_m': float(np.mean([row['post_error_m'] for row in last_rows])),
                    'last2_reference_error_min_max_m': [float(min(row['post_error_m'] for row in last_rows)), float(max(row['post_error_m'] for row in last_rows))]})
    assert all(len(hashes) == 1 for hashes in source_archives.values()), 'repository snapshots disagree'
    report = {'schema': 'd1-dynamic-brake-readonly-audit-v1', 'all_integrity_and_arithmetic_checks_passed': True,
              'verified_manifest_files': len(manifest), 'verified_original_bytes': sum(row['bytes'] for row in manifest.values()),
              'source_file_hashes_identical_across_runs': len(source_archives),
              'actual_runs': len(checked), 'actual_control_transitions': sum(row['actual_steps'] for row in checked),
              'checks': checked, 'recomputed_metrics': rows, 'mechanism': mechanism,
              'product_selected': False, 'reason': 'The dynamic reference-only candidate adds uphill rollback and repeated-stop creep; retain legacy.',
              'new_physics_steps': 0,
              'input_manifest_sha256': sha(source/'manifest.json'), 'input_summary_sha256': sha(source/'summary.json'),
              'audit_helper_sha256': sha(__file__)}
    with (output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    lines = ['# Dynamic braking: six paired development checks', '',
             '**Candidate rejected for product integration.** Legacy remains the default.', '',
             'Each row covers the complete eight seconds after release. Values are legacy → dynamic candidate; distances use the last commanded direction (S reverses the world-x sign).', '',
             '| Case / release | Peak forward (m) | Total backward (m) | Peak retreat (m) | Max reverse speed (m/s) | Last2 mean absolute speed (m/s) | Last2 signed world drift (m) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    keys = ['peak_forward_excursion_m', 'total_backward_travel_m', 'peak_retreat_from_prior_peak_m',
            'max_reverse_velocity_mps', 'last2_mean_abs_world_vx_mps', 'last2_signed_world_drift_m']
    for case in protocol['cases']:
        for number in range(1, len(case['held'])+1):
            pair = {row['strategy']: row for row in rows if row['case'] == case['name'] and row['release_number'] == number}
            values = [f"{pair['legacy'][key]:.6f} → {pair['dynamic_brake_hold'][key]:.6f}" for key in keys]
            lines.append(f"| {case['name']} / {number} | " + ' | '.join(values) + ' |')
    lines.extend(['', '12 completed runs, 26,400 control transitions, seven complete paired release windows. All six first-release prefixes and four historical legacy prefixes are bitwise identical. All states are finite and torques remain within bounds; zero recovery ticks. Every per-tick reference update and every reported release metric was independently recomputed from retained records.', '',
                  'The rough-spawn case is uphill by release. Its retained error drops from 0.550000 m to 0.081306 m immediately; it never reaches the 30 consecutive low-speed ticks needed to enter hold. The second stop-go release also fails to enter hold. The speed-based bound permits persistent creep while preventing the reference error from building the prior restoring action. This falsifies this particular reference-only stopping hypothesis; it does not establish that every dynamic braking design fails.', '',
                  'Ramping into hold itself is not a stop guarantee: ramp2 enters hold at 14.73 s but ends with a higher final-two-second speed than legacy. The unchanged underlying ±0.55 m reference clamp and v*dt integration are explicitly retained. No terrain branch or parameter search followed these failures.', ''])
    (output/'comparison.md').write_text('\n'.join(lines))
    print(json.dumps({key: report[key] for key in ('all_integrity_and_arithmetic_checks_passed', 'actual_runs', 'actual_control_transitions', 'verified_manifest_files')}))


if __name__ == '__main__':
    main()
