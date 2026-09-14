"""Fixed command-domain checks of the 85D heading task, without training or tuning.

External yaw torque uses the existing plant.step argument. An observer records
the actual xfrc_applied entering every real MuJoCo substep; pre-writing that array
before env.step would be erased by the unchanged plant and is not used here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy
from scripts.run_d1_heading_study import source_hashes
from scripts.run_d1_wheel_common_mean_study import ROOT, sha256, write_json, write_manifest
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig

SCHEMA = 'd1-heading-g1-domain-evaluation-v1'
PROTOCOL_SCHEMA = 'd1-heading-g1-domain-probe-v1'
PROTOCOL_SHA256 = 'cf5dbd51042bfafa163116f4a7fb2a990726d47c5a6000558f28aa6866a664fc'
DT = .01


def command_at_tick(command, tick):
    fraction = float(np.clip((tick-command['settle_ticks'])/command['ramp_ticks'], 0., 1.))
    forward = command['forward_target_mps']*fraction
    if command['stop_tick'] is not None and tick >= command['stop_tick']:
        forward = 0.
    pulse = command['yaw_pulse']
    yaw = pulse['user_yaw_rate_rps'] if pulse and pulse['start_tick'] <= tick < pulse['end_tick_exclusive'] else 0.
    return D1MotionCommand(forward_velocity_mps=forward, yaw_rate_rps=yaw,
                           clearance_m=command['height_m'])


def command_source(command):
    def scheduled(seconds):
        return command_at_tick(command, round(float(seconds)/DT))
    return scheduled


def prescribed_wrench(case, tick):
    pulse = case['external_wrench']
    if pulse and pulse['start_tick'] <= tick < pulse['end_tick_exclusive']:
        return np.asarray(pulse['force_xyz_n']+pulse['torque_xyz_nm'], dtype=np.float64)
    return np.zeros(6, dtype=np.float64)


def phase_name(case, tick):
    command = case['command']
    pulse = command['yaw_pulse']
    if pulse:
        return 'before_turn' if tick < pulse['start_tick'] else 'turn' if tick < pulse['end_tick_exclusive'] else 'heading_hold'
    wrench = case['external_wrench']
    if wrench and tick >= wrench['start_tick']:
        return 'external_impulse' if tick < wrench['end_tick_exclusive'] else 'recovery'
    if command['stop_tick'] is not None and tick >= command['stop_tick']:
        return 'stopping'
    if tick < command['settle_ticks']:
        return 'settling'
    return 'ramp' if tick < command['settle_ticks']+command['ramp_ticks'] else 'driving'


class NativeWrenchObserver:
    """Instance step bridge and read-only MuJoCo-entry instrumentation.

    This probe runs episodes sequentially in its own process. The module-level
    mj_step observer is restored on every exit and never changes its arguments.
    """

    def __init__(self, plant, case):
        self.plant, self.case = plant, case
        self.samples = []
        self._original_step = plant.step
        self._original_mj_step = mujoco.mj_step

    def __enter__(self):
        def native_step(torque, **kwargs):
            wrench = prescribed_wrench(self.case, round(float(self.plant.data.time)/DT))
            if kwargs.get('push_torque_world_nm') is not None:
                raise RuntimeError('probe refuses to overwrite another torque perturbation')
            if np.any(wrench[:3]):
                raise ValueError('frozen G1 cases contain only yaw torque, no force')
            # All existing arguments, including the loop's force=None, survive.
            if np.any(wrench[3:]):
                kwargs['push_torque_world_nm'] = wrench[3:].copy()
            return self._original_step(torque, **kwargs)

        def observed_mj_step(model, data, *args, **kwargs):
            if model is self.plant.model and data is self.plant.data:
                if args or kwargs:
                    raise RuntimeError('expected exactly one native physics substep')
                before = float(data.time)
                actual = data.xfrc_applied[self.plant.base_body_id].copy()
                other = data.xfrc_applied.copy()
                other[self.plant.base_body_id] = 0.
                if np.any(other):
                    raise RuntimeError('unexpected wrench on a different body')
                result = self._original_mj_step(model, data)
                self.samples.append({'start_time_s': before,
                                     'actual_dt_s': float(data.time)-before,
                                     'wrench_world': actual.tolist()})
                return result
            return self._original_mj_step(model, data, *args, **kwargs)

        self.plant.step = native_step
        mujoco.mj_step = observed_mj_step
        return self

    def __exit__(self, _type, _value, _traceback):
        self.plant.step = self._original_step
        mujoco.mj_step = self._original_mj_step
        self.plant.data.xfrc_applied[:] = 0.


def rmse(values):
    return math.sqrt(float(np.mean(np.square(values))))


def endpoint_window(rows, start, end):
    selected = [row for row in rows if start <= row['endpoint_tick'] < end]
    return selected if len(selected) == end-start else None


def window_max(rows, start, end, key):
    selected = endpoint_window(rows, start, end)
    return None if selected is None else float(max(abs(row[key]) for row in selected))


def action_statistics(rows):
    action = np.asarray([row['applied_action'] for row in rows])
    return {'mean_by_channel': action.mean(axis=0).tolist(),
            'min_by_channel': action.min(axis=0).tolist(),
            'max_by_channel': action.max(axis=0).tolist(),
            'box_boundary_occupancy_by_channel': (np.abs(action) >= 1.-1e-6).mean(axis=0).tolist(),
            'near_boundary_0p98_occupancy_by_channel': (np.abs(action) >= .98).mean(axis=0).tolist()}


def aggregate_rows(rows):
    return {'actual_transitions': len(rows),
            'actual_seconds': sum(row['heading_task']['actual_dt_s'] for row in rows),
            'velocity_rmse_mps': rmse([row['user_forward_error_mps'] for row in rows]),
            'user_yaw_rate_rmse_rps': rmse([row['heading_task']['user_yaw_rate_error_after'] for row in rows]),
            'height_rmse_m': rmse([row['metrics']['height_error_m'] for row in rows]),
            'heading_rmse_rad': rmse([row['heading_error_rad'] for row in rows]),
            'heading_peak_rad': max(abs(row['heading_error_rad']) for row in rows),
            'cross_track_rmse_m': rmse([row['cross_track_after_m'] for row in rows]),
            'cross_track_peak_m': max(abs(row['cross_track_after_m']) for row in rows),
            'cross_track_final_m': rows[-1]['cross_track_after_m'],
            'task_return': sum(row['reward'] for row in rows),
            'servo_return': sum(row['servo_reward'] for row in rows),
            'heading_return': sum(row['reward_terms']['heading_goal'] for row in rows),
            'reward_terms_return': {key: sum(row['reward_terms'][key] for row in rows) for key in rows[0]['reward_terms']},
            'actions': action_statistics(rows)}


def gates_and_metrics(case, rows, positions, thresholds):
    """Exact endpoint windows; incomplete windows fail instead of becoming zero."""
    metric = aggregate_rows(rows)
    last = rows[-1]
    metric.update(
        completed_duration=bool(last['truncated'] and not last['terminated'] and len(rows) == case['max_transitions']),
        nonwheel_contact_steps=sum(row['metrics']['undesired_ground_contacts'] > 0 for row in rows),
        max_abs_actual_roll_pitch_rad=max(max(abs(v) for v in row['actual_roll_pitch_rad']) for row in rows),
        max_abs_ground_relative_roll_pitch_rad=max(max(abs(row['metrics'][key]) for key in ('roll_error_rad', 'pitch_error_rad')) for row in rows),
        planar_origin_displacement_peak_m=float(np.max(np.linalg.norm(positions[:, :2]-positions[0, :2], axis=1))),
        terminal_reason=last['terminal_reason'],
        final_reference_delta_rad=math.atan2(math.sin(last['heading_task']['reference_heading_after']-rows[0]['heading_task']['reference_heading_before']), math.cos(last['heading_task']['reference_heading_after']-rows[0]['heading_task']['reference_heading_before'])),
    )

    def maximum(value, limit):
        return value is not None and value <= limit

    gate = thresholds['all_cases']
    checks = {
        'complete_duration': metric['completed_duration'],
        'no_nonwheel_contact': metric['nonwheel_contact_steps'] <= gate['nonwheel_contact_steps_max'],
        'actual_roll_pitch': metric['max_abs_actual_roll_pitch_rad'] <= gate['max_abs_actual_roll_pitch_rad'],
        'height_rmse': metric['height_rmse_m'] <= gate['height_rmse_m_max'],
        'heading_peak': metric['heading_peak_rad'] <= gate['heading_peak_rad_max'],
        'no_boundary_or_fall': not last['terminated'],
    }
    if 'forward_and_impulse' in case['gate_groups']:
        gate = thresholds['forward_and_impulse']
        checks.update(velocity_rmse=metric['velocity_rmse_mps'] <= gate['velocity_rmse_mps_max'],
                      heading_rmse=metric['heading_rmse_rad'] <= gate['heading_rmse_rad_max'],
                      cross_track_peak=metric['cross_track_peak_m'] <= gate['cross_track_peak_m_max'],
                      cross_track_final=abs(metric['cross_track_final_m']) <= gate['cross_track_final_abs_m_max'])
    if 'stop_cases' in case['gate_groups']:
        gate = thresholds['stop_cases']
        metric['late_stop_max_abs_body_vx_mps'] = window_max(rows, gate['speed_window_start_tick'], gate['speed_window_end_tick_exclusive'], 'body_forward_mps')
        first, end = gate['stationary_path_window_start_tick'], gate['stationary_path_window_end_tick_exclusive']
        metric['late_stop_cumulative_planar_path_m'] = None if len(positions) <= end else float(np.linalg.norm(np.diff(positions[first:end+1, :2], axis=0), axis=1).sum())
        release = gate['release_tick']
        stop_metrics = None
        if len(positions) > release+1:
            direction = float(np.sign(case['command']['forward_target_mps']))
            progress = direction*(positions[release:, 0]-positions[release, 0])
            delta = np.diff(progress)
            stop_metrics = {'observed_transitions': len(delta),
                'net_world_x_m': float(positions[-1, 0]-positions[release, 0]),
                'peak_forward_excursion_m': float(max(0., np.max(progress))),
                'max_backward_excursion_m': float(max(0., -np.min(progress))),
                'total_backward_travel_m': float(np.maximum(-delta, 0.).sum()),
                'peak_retreat_from_prior_peak_m': float(np.max(np.maximum.accumulate(progress)-progress)),
                'max_reverse_world_speed_mps': float(max(0., -np.min(delta/DT)))}
        metric['stop_motion'] = stop_metrics
        checks.update(late_stop_speed=maximum(metric['late_stop_max_abs_body_vx_mps'], gate['max_abs_body_vx_late_mps']),
                      late_stop_planar_path=maximum(metric['late_stop_cumulative_planar_path_m'], gate['cumulative_planar_path_after_first_second_m_max']),
                      heading_rmse=metric['heading_rmse_rad'] <= gate['heading_rmse_rad_max'],
                      velocity_rmse=metric['velocity_rmse_mps'] <= gate['velocity_rmse_mps_max'])
    if 'turn_cases' in case['gate_groups']:
        gate = thresholds['turn_cases']
        first, end = gate['hold_settling_window_start_tick'], gate['hold_window_end_tick_exclusive']
        metric['turn_late_max_abs_heading_error_rad'] = window_max(rows, first, end, 'heading_error_rad')
        metric['turn_late_max_abs_body_vx_mps'] = window_max(rows, first, end, 'body_forward_mps')
        target = math.copysign(gate['expected_final_reference_delta_rad'], case['command']['yaw_pulse']['user_yaw_rate_rps'])
        checks.update(turn_reference=abs(metric['final_reference_delta_rad']-target) < 1e-10,
                      turn_late_heading=maximum(metric['turn_late_max_abs_heading_error_rad'], gate['max_abs_heading_error_after_settle_rad']),
                      turn_late_speed=maximum(metric['turn_late_max_abs_body_vx_mps'], gate['max_abs_body_vx_after_settle_mps']),
                      turn_planar_displacement=metric['planar_origin_displacement_peak_m'] <= gate['planar_origin_displacement_peak_m_max'])
    if 'impulse_cases' in case['gate_groups']:
        gate = thresholds['impulse_cases']
        metric['impulse_recovery_max_abs_heading_error_rad'] = window_max(rows, gate['recovery_window_start_tick'], gate['recovery_window_end_tick_exclusive'], 'heading_error_rad')
        checks['impulse_recovery'] = maximum(metric['impulse_recovery_max_abs_heading_error_rad'], gate['max_abs_heading_error_after_recovery_rad'])
    return metric, {'checks': checks, 'passed': all(checks.values()), 'failed': [name for name, ok in checks.items() if not ok]}


def run_episode(output, case, model_spec, thresholds):
    output.mkdir(parents=True, exist_ok=False)
    env = D1HeadingTrackingEnv(terrain=D1LocomotionTerrainConfig(**case['terrain']),
                              episode_seconds=case['episode_seconds'], command_source=command_source(case['command']))
    rows, qpos, qvel, observations, truth_positions = [], [], [], [], []
    try:
        observation, metadata = env.reset(seed=case['seed'])
        write_json(output/'episode_metadata.json', metadata)
        policy = None if model_spec['kind'] == 'zero_action' else load_heading_policy(model_spec['checkpoint'], model_spec['metadata'], env)
        if policy is not None and policy.num_timesteps != 65536:
            raise ValueError('only the declared final 65536-step policies are accepted')
        initial_position = env.plant.base_position.copy()
        qpos.append(env.plant.data.qpos.copy())
        qvel.append(env.plant.data.qvel.copy())
        observations.append(observation.copy())
        truth_positions.append(initial_position)
        initial_hash = hashlib.sha256(qpos[0].tobytes()+qvel[0].tobytes()+observations[0].tobytes()).hexdigest()
        with NativeWrenchObserver(env.plant, case) as observer, gzip.open(output/'trace.jsonl.gz', 'xt') as stream:
            for tick in range(env.max_steps):
                if round(float(env.plant.data.time)/DT) != tick:
                    raise RuntimeError('unexpected control time or hidden reset')
                expected_command = command_at_tick(case['command'], tick)
                action = np.zeros(8, dtype=np.float32) if policy is None else policy.predict(observation, deterministic=True)[0]
                start = len(observer.samples)
                observation, reward, terminated, truncated, info = env.step(action)
                samples = observer.samples[start:]
                expected_wrench = prescribed_wrench(case, tick)
                if len(samples) != env.plant.physics_steps or any(not np.array_equal(np.asarray(sample['wrench_world']), expected_wrench) for sample in samples):
                    raise RuntimeError('actual physics-entry wrench differs from protocol')
                if info['heading_task']['user_command_before'] != asdict(expected_command):
                    raise RuntimeError('executed user command differs from fixed schedule')
                if not np.array_equal(np.asarray(action), info['applied_action']):
                    raise RuntimeError('unexpected action transformation')
                truth = env.last_transition.truth
                quat = env.plant.data.qpos[3:7]
                w, x, y, z = quat
                actual_roll_pitch = [float(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
                                     float(np.arcsin(np.clip(2*(w*y-z*x), -1., 1.)))]
                if not np.allclose(actual_roll_pitch, truth.base_rpy[:2], atol=1e-12, rtol=0.):
                    raise RuntimeError('quaternion attitude differs from actual truth')
                row = {'tick': tick, 'endpoint_tick': tick+1, 'phase': phase_name(case, tick),
                    'heading_task': info['heading_task'], 'reward': reward, 'reward_terms': info['reward_terms'],
                    'servo_reward': info['servo_reward'], 'servo_reward_terms': info['servo_reward_terms'],
                    'metrics': info['metrics'], 'terrain_exposure': info['terrain_exposure'],
                    'body_forward_mps': float(truth.base_linear_velocity_body[0]),
                    'user_forward_error_mps': float(truth.base_linear_velocity_body[0]-expected_command.forward_velocity_mps),
                    'heading_error_rad': info['heading_task']['heading_error_after'],
                    'actual_roll_pitch_rad': actual_roll_pitch,
                    'truth_position_world_m': truth.base_position.tolist(),
                    'cross_track_after_m': float(truth.base_position[1]-initial_position[1]),
                    'requested_action': np.asarray(action).tolist(), 'applied_action': info['applied_action'].tolist(),
                    'physics_wrench_samples': samples,
                    'actual_signed_yaw_impulse_nms': sum(sample['wrench_world'][5]*sample['actual_dt_s'] for sample in samples),
                    'terminated': bool(terminated), 'truncated': bool(truncated), 'terminal_reason': info['terminal_reason']}
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                rows.append(row)
                qpos.append(env.plant.data.qpos.copy())
                qvel.append(env.plant.data.qvel.copy())
                observations.append(observation.copy())
                truth_positions.append(truth.base_position.copy())
                if terminated or truncated:
                    break
        positions = np.asarray(truth_positions)
        np.savez_compressed(output/'states.npz', qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                            observations=np.asarray(observations), truth_positions_world_m=positions,
                            requested_actions=np.asarray([row['requested_action'] for row in rows]),
                            applied_actions=np.asarray([row['applied_action'] for row in rows]))
        metrics, gates = gates_and_metrics(case, rows, positions, thresholds)
        impulse = sum(row['actual_signed_yaw_impulse_nms'] for row in rows)
        expected_impulse = sum(prescribed_wrench(case, tick)[5]*DT for tick in range(len(rows)))
        if abs(impulse-expected_impulse) > 1e-10:
            raise RuntimeError('physical impulse integral differs from executed protocol intervals')
        final_position = positions[-1]
        safe = np.asarray(metadata['safe_half_size_m'])
        summary = {'case': case['name'], 'model': model_spec['label'], 'initial_state_observation_sha256': initial_hash,
            **metrics, 'gates': gates, 'phase_metrics': {phase: aggregate_rows([row for row in rows if row['phase'] == phase]) for phase in dict.fromkeys(row['phase'] for row in rows)},
            'actual_signed_yaw_impulse_nms': impulse, 'expected_observed_yaw_impulse_nms': expected_impulse,
            'actual_perturbed_control_intervals': sum(any(sample['wrench_world'][5] != 0. for sample in row['physics_wrench_samples']) for row in rows),
            'actual_observed_physics_substeps': sum(len(row['physics_wrench_samples']) for row in rows),
            'wrench_zero_after_episode': bool(np.all(env.plant.data.xfrc_applied == 0.)),
            'terminal_state': {'x_boundary': bool(abs(final_position[0]) >= safe[0]),
                               'y_boundary': bool(abs(final_position[1]) >= safe[1]),
                               'clearance_m': rows[-1]['metrics']['clearance_m'],
                               'nonwheel_contacts': rows[-1]['metrics']['undesired_ground_contacts'],
                               'actual_roll_pitch_rad': rows[-1]['actual_roll_pitch_rad']},
            'classification': 'fixed development command-domain probe; not unseen-terrain generalization',
            'model_selected_or_training_performed': False}
        write_json(output/'summary.json', summary)
        write_manifest(output)
        return summary, rows
    except BaseException as error:
        write_json(output/'execution_failure.json', {'type': type(error).__name__, 'message': str(error),
                                                     'fully_recorded_transitions': len(rows)})
        raise
    finally:
        env.plant.data.xfrc_applied[:] = 0.
        env.close()


def validated_inputs(protocol_path, models_root=None):
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('G1 protocol hash differs from the reviewed revision 2')
    protocol = json.loads(protocol_path.read_text())
    if protocol['schema'] != PROTOCOL_SCHEMA:
        raise ValueError('unknown G1 protocol schema')
    inputs = {str(ROOT/name): value for name, value in source_hashes().items()}
    inputs[str(Path(__file__).resolve())] = sha256(__file__)
    inputs[str(protocol_path.resolve())] = sha256(protocol_path)
    for item in protocol['models']:
        if item['kind'] == 'zero_action':
            continue
        if models_root is not None:
            base = models_root/f"seed{item['seed']}/checkpoints/step65536/model.zip"
            item['checkpoint'], item['metadata'] = str(base), str(base.with_suffix('.metadata.json'))
        for field, hash_key in (('checkpoint', 'model_sha256'), ('metadata', 'metadata_sha256')):
            path = Path(item[field]).resolve()
            if sha256(path) != item[hash_key]:
                raise ValueError(f'declared {field} hash does not match: {path}')
            inputs[str(path)] = item[hash_key]
        extra = json.loads(Path(item['metadata']).read_text())['extra']
        if (extra['pipeline_smoke_only'] is not False or extra['saved_after_completed_update'] is not True
                or extra['actual_transitions'] != 65536 or extra['checkpoint_budget'] != 65536
                or extra['seed'] != item['seed']):
            raise ValueError('checkpoint is not the declared complete final training checkpoint')
    return protocol, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models-root', type=Path)
    parser.add_argument('--mode', choices=('evaluate', 'interface-smoke'), default='evaluate')
    args = parser.parse_args(argv)
    protocol, inputs = validated_inputs(args.protocol, args.models_root)
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cases = protocol['cases']
    models = protocol['models']
    if args.mode == 'interface-smoke':
        # Explicit separate interface check: two real intervals, no trained model.
        case = json.loads(json.dumps(cases[4]))
        case.update(name='positive_torque_interface_smoke_2ticks', episode_seconds=.02,
                    max_transitions=2, gate_groups=['all_cases'], seed=55999)
        case['external_wrench'].update(start_tick=0, end_tick_exclusive=2)
        cases, models = [case], [models[0]]
    write_json(args.output/'protocol.json', {'schema': SCHEMA, 'mode': args.mode,
        'reviewed_protocol_sha256': PROTOCOL_SHA256, 'cases': cases, 'models': models,
        'input_sha256': inputs, 'max_physical_transitions': sum(case['max_transitions'] for case in cases)*len(models),
        'torch_num_threads': torch.get_num_threads(), 'failure_rule': 'first termination; no resets or padding',
        'interface_smoke_excluded_from_formal_matrix': True})
    summaries, common = [], []
    for case in cases:
        zero_trace, zero_hash = None, None
        for model in models:
            summary, rows = run_episode(args.output/case['name']/model['label'], case, model, protocol['proposed_gates'])
            summaries.append(summary)
            if model['kind'] == 'zero_action':
                zero_trace, zero_hash = rows, summary['initial_state_observation_sha256']
            else:
                if zero_trace is None or summary['initial_state_observation_sha256'] != zero_hash:
                    raise RuntimeError('each model requires a same-initial-state zero comparison')
                count = min(len(zero_trace), len(rows))
                if any(zero_trace[i]['heading_task']['user_command_before'] != rows[i]['heading_task']['user_command_before'] for i in range(count)):
                    raise RuntimeError('model and zero received different common-prefix commands')
                common.append({'case': case['name'], 'model': model['label'], 'actual_common_transitions': count,
                               'zero': aggregate_rows(zero_trace[:count]), 'policy': aggregate_rows(rows[:count]),
                               'definition': 'strict shared actually observed prefix; no filling missing steps'})
            print(json.dumps({'case': case['name'], 'model': model['label'], 'steps': summary['actual_transitions'],
                              'g1_pass': summary['gates']['passed'], 'failed': summary['gates']['failed'],
                              'velocity_rmse': summary['velocity_rmse_mps'], 'heading_peak': summary['heading_peak_rad'],
                              'yaw_impulse_nms': summary['actual_signed_yaw_impulse_nms']}, allow_nan=False), flush=True)
    changed = [name for name, expected in inputs.items() if sha256(name) != expected]
    if changed:
        raise RuntimeError(f'probe inputs changed during execution: {changed}')
    write_json(args.output/'common_prefix.json', common)
    write_json(args.output/'summary.json', {'schema': SCHEMA, 'mode': args.mode, 'input_unchanged': True,
        'actual_episodes': len(summaries), 'actual_control_transitions': sum(row['actual_transitions'] for row in summaries),
        'episodes': summaries, 'no_training_or_model_selection': True})
    write_manifest(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
