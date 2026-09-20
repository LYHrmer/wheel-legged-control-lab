"""Independent read-only audit of the fixed release experiment; no plant created."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads((args.input / 'protocol.json').read_text())
    summary = json.loads((args.input / 'summary.json').read_text())
    report = {'new_physics_steps': 0, 'plant_instantiated': False, 'episodes': [], 'pairs': []}
    total = substeps = 0
    for case in protocol['cases']:
        pair = []
        for condition in protocol['conditions']:
            path = args.input / case['name'] / condition
            rows = read_rows(path / 'trace.jsonl.gz')
            s = json.loads((path / 'summary.json').read_text())
            prepared = json.loads((path / 'prepared_commands.json').read_text())
            with np.load(path / 'states.npz') as data:
                states = {k: data[k].copy() for k in data.files}
            count = len(rows)
            assert count == case['max_transitions']
            assert len(prepared) == count + 1
            assert sum(p['physically_executed'] for p in prepared) == count
            assert all(p['prepare_succeeded'] for p in prepared)
            assert states['qpos'].shape == (count + 1, 23)
            assert states['qvel'].shape == (count + 1, 22)
            assert states['observations'].dtype == np.float32
            assert np.all(states['applied_actions'] == 0)
            vx = np.array([row['body_forward_mps'] for row in rows])
            raw = np.array([row['raw_user_command']['forward_velocity_mps'] for row in rows])
            served = np.array([row['served_reference_command']['forward_velocity_mps'] for row in rows])
            expected = case['command']['forward_target_mps'] * np.clip((np.arange(count)-50)/50, 0, 1)
            if case['command']['stop_tick'] is not None:
                expected[400:] = 0
            assert same(raw, expected)
            assert same(vx-raw, np.array([row['user_forward_error_mps'] for row in rows]))
            rms = float(np.sqrt(np.mean((vx-raw)**2)))
            assert rms == s['velocity_rmse_mps']
            assert np.array_equal(vx.astype(np.float32), states['observations'][1:, 0])
            for k, row in enumerate(rows):
                assert row['tick'] == k and row['endpoint_tick'] == k+1
                assert row['raw_user_command'] == prepared[k]['raw_command']
                assert row['served_reference_command'] == prepared[k]['served_command']
                assert row['heading_task']['user_command_before'] == prepared[k]['served_command']
                assert row['heading_task']['user_command_semantics'] == 'served_reference_not_raw_operator_intent'
                assert len(row['physics_wrench_samples']) == 5
                assert len(row['actuator_applied_nm']) == 5
                assert row['raw_user_command']['yaw_rate_rps'] == row['served_reference_command']['yaw_rate_rps']
                assert row['raw_user_command']['clearance_m'] == row['served_reference_command']['clearance_m']
            impulse = sum(sample['wrench_world'][5]*sample['actual_dt_s']
                          for row in rows for sample in row['physics_wrench_samples'])
            target_impulse = case['external_wrench']['torque_xyz_nm'][2]*.2 if case['external_wrench'] else 0
            assert abs(impulse-target_impulse) < 1e-10
            item = {'case': case['name'], 'condition': condition,
                    'raw_goal_velocity_rmse_mps': rms,
                    'served_reference_velocity_rmse_mps': float(np.sqrt(np.mean((vx-served)**2))),
                    'gates': s['gates'], 'transitions': count, 'source_records': len(prepared)}
            if case['command']['stop_tick'] is not None:
                late_speed = float(np.max(np.abs(vx[499:799])))
                positions = states['truth_positions_world_m']
                late_path = float(np.linalg.norm(np.diff(positions[500:701, :2], axis=0), axis=1).sum())
                assert late_speed == s['late_stop_max_abs_body_vx_mps']
                assert late_path == s['late_stop_cumulative_planar_path_m']
                before_vx = np.array([row['body_forward_before_mps'] for row in rows])
                wheel = states['qvel'][:-1, 6:][:, [3, 7, 11, 15]].mean(axis=1)*.087
                assert same(wheel, np.array([row['wheel_rolling_before_mps'] for row in rows]))
                mismatch = float(np.sqrt(np.mean((before_vx[400:450]-wheel[400:450])**2)))
                assert mismatch == s['release_diagnostics']['first_0p5s_body_wheel_mismatch_rms_mps']
                progress = np.sign(case['command']['forward_target_mps']) * (positions[400:, 0]-positions[400, 0])
                motion = {'peak_forward_excursion_m': float(max(0., progress.max())),
                          'peak_retreat_from_prior_peak_m': float(np.max(np.maximum.accumulate(progress)-progress)),
                          'total_backward_travel_m': float(np.maximum(-np.diff(progress), 0).sum()),
                          'net_world_x_m': float(positions[-1, 0]-positions[400, 0])}
                assert all(value == s['stop_motion'][key] for key, value in motion.items())
                item.update(late_speed_mps=late_speed, late_path_m=late_path, stop_motion=motion,
                            first_0p5s_mismatch_rms_mps=mismatch,
                            wheel_mean_requested_torque_at399_400_nm=[float(np.mean(rows[k]['requested_torque_nm'][3::4])) for k in (399, 400)],
                            wheel_mean_applied_torque_at400_nm=float(np.mean(np.asarray(rows[400]['actuator_applied_nm'])[:, [3, 7, 11, 15]])),
                            release_timing=s['release_diagnostics'])
            pair.append(states)
            report['episodes'].append(item)
            total += count
            substeps += sum(len(row['physics_wrench_samples']) for row in rows)
        stop = case['command']['stop_tick']
        checks = {}
        for key in pair[0]:
            n = None if stop is None else 401 if key in ('qpos', 'qvel', 'truth_positions_world_m') else 400
            checks[key] = same(pair[0][key][:n], pair[1][key][:n])
        assert all(checks.values())
        report['pairs'].append({'case': case['name'], 'bitwise_checks': checks})
    assert total == summary['actual_control_transitions'] == 8000
    assert substeps == summary['actual_physics_substeps'] == 40000
    assert summary['comparison_valid'] and not summary['candidate_passed_both_stop_cases']
    for relative, spec in json.loads((args.input / 'manifest.json').read_text()).items():
        path = args.input / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
    report.update(audit_checks_passed=True, actual_control_transitions=total,
                  actual_physics_substeps=substeps, candidate_passed=False,
                  conclusion='Fixed release_0p5 fails both original stop cases; keep old behavior.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('episodes', 'pairs')}))


if __name__ == '__main__':
    main()
