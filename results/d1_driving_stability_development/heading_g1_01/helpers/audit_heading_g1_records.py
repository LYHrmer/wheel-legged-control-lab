"""Independently verify archived G1 data and arithmetic, without stepping physics."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrapped(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.input/'manifest.json').read_text())
    for name, expected in manifest.items():
        path = args.input/name
        assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], name
    protocol = json.loads(args.protocol.read_text())
    result = json.loads((args.input/'summary.json').read_text())
    captured = json.loads((args.input/'protocol.json').read_text())
    assert digest(args.protocol) == captured['reviewed_protocol_sha256']
    by_case = {case['name']: case for case in protocol['cases']}
    initial, checks, all_traces = {}, [], {}
    for episode in result['episodes']:
        case, model = episode['case'], episode['model']
        folder = args.input/case/model
        with gzip.open(folder/'trace.jsonl.gz', 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        with np.load(folder/'states.npz', allow_pickle=False) as state:
            pos, obs = state['qpos'], state['observations']
            actions, requested = state['applied_actions'], state['requested_actions']
            truth_position = state['truth_positions_world_m']
            initial_hash = hashlib.sha256(pos[0].tobytes()+state['qvel'][0].tobytes()+obs[0].tobytes()).hexdigest()
        assert initial_hash == episode['initial_state_observation_sha256']
        if case in initial:
            assert initial_hash == initial[case]
        initial[case] = initial_hash
        spec = by_case[case]
        assert len(rows) == spec['max_transitions'] == episode['actual_transitions']
        assert obs.shape == (len(rows)+1, 85) and actions.shape == (len(rows), 8)
        assert np.isfinite(pos).all() and np.isfinite(obs).all() and np.isfinite(actions).all()
        assert np.array_equal(actions, requested) and np.all(np.abs(actions) <= 1.)
        assert np.allclose(pos[:, :3], truth_position, atol=1e-12, rtol=0.)
        errors, speeds, heights, tilt, impulse = [], [], [], [], 0.
        perturbed = substeps = 0
        for tick, row in enumerate(rows):
            assert row['tick'] == tick and row['endpoint_tick'] == tick+1
            command = spec['command']
            fraction = max(0., min(1., (tick-command['settle_ticks'])/command['ramp_ticks']))
            forward = command['forward_target_mps']*fraction
            if command['stop_tick'] is not None and tick >= command['stop_tick']:
                forward = 0.
            pulse = command['yaw_pulse']
            yaw = pulse['user_yaw_rate_rps'] if pulse and pulse['start_tick'] <= tick < pulse['end_tick_exclusive'] else 0.
            task = row['heading_task']
            user = task['user_command_before']
            assert user['forward_velocity_mps'] == forward and user['yaw_rate_rps'] == yaw
            assert user['clearance_m'] == command['height_m']
            assert abs(task['reference_heading_after']-wrapped(task['reference_heading_before']+task['actual_dt_s']*yaw)) < 1e-12
            w, x, y, z = pos[tick+1, 3:7]
            actual_yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            actual_tilt = [math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                           math.asin(max(-1., min(1., 2*(w*y-z*x))))]
            error = wrapped(actual_yaw-task['reference_heading_after'])
            assert abs(error-row['heading_error_rad']) < 1e-12
            assert np.allclose(actual_tilt, row['actual_roll_pitch_rad'], atol=1e-12, rtol=0.)
            assert abs(float(obs[tick+1, 0])-row['body_forward_mps']) < 1e-7
            assert row['metrics']['ground_height_m'] == 0.
            height = pos[tick+1, 2]-command['height_m']
            assert abs(height-row['metrics']['height_error_m']) < 1e-12
            assert abs(row['body_forward_mps']-forward-row['user_forward_error_mps']) < 1e-12
            heading_reward = task['actual_dt_s']*.5*(math.exp(-(error/protocol['environment']['heading_sigma_rad'])**2)-1.)
            assert abs(heading_reward-row['reward_terms']['heading_goal']) < 1e-12
            assert abs(sum(row['reward_terms'].values())-row['reward']) < 1e-12
            assert abs(sum(row['servo_reward_terms'].values())-row['servo_reward']) < 1e-12
            assert np.array_equal(actions[tick], row['applied_action'])
            expected = [0.]*6
            pulse = spec['external_wrench']
            if pulse and pulse['start_tick'] <= tick < pulse['end_tick_exclusive']:
                expected = pulse['force_xyz_n']+pulse['torque_xyz_nm']
                perturbed += 1
            samples = row['physics_wrench_samples']
            assert len(samples) == 5
            control_impulse = 0.
            for sample in samples:
                assert sample['wrench_world'] == expected
                assert abs(sample['actual_dt_s']-.002) < 1e-12
                control_impulse += sample['actual_dt_s']*sample['wrench_world'][5]
            assert abs(control_impulse-row['actual_signed_yaw_impulse_nms']) < 1e-12
            impulse += control_impulse
            substeps += len(samples)
            errors.append(error)
            heights.append(height)
            speeds.append(row['body_forward_mps']-forward)
            tilt.extend(actual_tilt)
            assert not row['terminated']
            assert row['truncated'] == (tick == len(rows)-1)
            assert not row['terrain_exposure']['nonflat_now']
        rms = lambda data: math.sqrt(float(np.mean(np.square(data))))
        for key, value in {'velocity_rmse_mps': rms(speeds), 'height_rmse_m': rms(heights),
                           'heading_rmse_rad': rms(errors), 'heading_peak_rad': max(abs(v) for v in errors),
                           'max_abs_actual_roll_pitch_rad': max(abs(v) for v in tilt)}.items():
            assert abs(value-episode[key]) < 1e-12, key
        expected_impulse = 0. if spec['external_wrench'] is None else math.copysign(.1, spec['external_wrench']['torque_xyz_nm'][2])
        assert abs(impulse-expected_impulse) < 1e-10
        assert episode['actual_observed_physics_substeps'] == substeps
        assert episode['actual_perturbed_control_intervals'] == perturbed
        assert episode['wrench_zero_after_episode']
        if 'stop_cases' in spec['gate_groups']:
            body = np.asarray([row['body_forward_mps'] for row in rows])
            # Endpoints500..799 are rows499..798; position path includes500..700.
            peak = float(np.abs(body[499:799]).max())
            path = float(np.linalg.norm(np.diff(truth_position[500:701, :2], axis=0), axis=1).sum())
            assert peak == episode['late_stop_max_abs_body_vx_mps']
            assert abs(path-episode['late_stop_cumulative_planar_path_m']) < 1e-12
        if 'turn_cases' in spec['gate_groups']:
            peak_error = max(abs(value) for value in errors[449:799])
            peak_speed = max(abs(row['body_forward_mps']) for row in rows[449:799])
            assert abs(peak_error-episode['turn_late_max_abs_heading_error_rad']) < 1e-12
            assert peak_speed == episode['turn_late_max_abs_body_vx_mps']
            assert abs(abs(episode['final_reference_delta_rad'])-.3) < 1e-10
        if 'impulse_cases' in spec['gate_groups']:
            peak = max(abs(value) for value in errors[819:1199])
            assert abs(peak-episode['impulse_recovery_max_abs_heading_error_rad']) < 1e-12
        checks.append({'case': case, 'model': model, 'actual_transitions': len(rows),
                       'physics_substeps': substeps, 'actual_yaw_impulse_nms': impulse,
                       'all_record_and_metric_checks_passed': True})
        all_traces[(case, model)] = rows
    common = json.loads((args.input/'common_prefix.json').read_text())
    assert len(common) == 18
    for pair in common:
        zero = all_traces[(pair['case'], 'zero')]
        learned = all_traces[(pair['case'], pair['model'])]
        count = min(len(zero), len(learned))
        assert count == pair['actual_common_transitions']
        assert [row['heading_task']['user_command_before'] for row in zero[:count]] == [row['heading_task']['user_command_before'] for row in learned[:count]]
        for label, trace in (('zero', zero), ('policy', learned)):
            value = math.sqrt(float(np.mean([row['user_forward_error_mps']**2 for row in trace[:count]])))
            assert abs(value-pair[label]['velocity_rmse_mps']) < 1e-12
    report = {'schema': 'd1-heading-g1-independent-record-audit-v1', 'all_passed': True,
              'actual_episodes': len(checks), 'actual_control_transitions': sum(row['actual_transitions'] for row in checks),
              'actual_physics_substeps': sum(row['physics_substeps'] for row in checks),
              'verified_manifest_files': len(manifest), 'strict_common_prefix_pairs': len(common),
              'checks': checks, 'new_physics_steps': 0,
              'manifest_sha256': digest(args.input/'manifest.json'),
              'protocol_sha256': digest(args.protocol), 'helper_sha256': digest(__file__)}
    with (args.output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('all_passed', 'actual_episodes', 'actual_control_transitions', 'actual_physics_substeps', 'strict_common_prefix_pairs')}))


if __name__ == '__main__':
    main()
