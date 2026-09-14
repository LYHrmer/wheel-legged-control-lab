"""Six frozen, work-only paired dynamic braking checks through the real GUI runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from dynamic_brake import DynamicBrakeHold


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


CASES = [
    {'name': 'flat3', 'arena': 'flat', 'zone': 'start', 'gear': 3, 'key': 'W',
     'held': [[2., 10.]], 'duration': 18., 'old_name': 'flat_start_gear3'},
    {'name': 'ramp2', 'arena': 'course', 'zone': 'ramp', 'gear': 2, 'key': 'W',
     'held': [[2., 14.]], 'duration': 22., 'old_name': 'course_ramp_gear2'},
    {'name': 'rough3', 'arena': 'course', 'zone': 'rough', 'gear': 3, 'key': 'W',
     'held': [[2., 14.]], 'duration': 22., 'old_name': 'course_rough_gear3'},
    {'name': 'stairs3', 'arena': 'course', 'zone': 'stairs', 'gear': 3, 'key': 'W',
     'held': [[2., 14.]], 'duration': 22., 'old_name': 'course_stairs_gear3'},
    {'name': 'flat_reverse3', 'arena': 'flat', 'zone': 'start', 'gear': 3, 'key': 'S',
     'held': [[2., 10.]], 'duration': 18., 'old_name': None},
    {'name': 'flat_stop_go3', 'arena': 'flat', 'zone': 'start', 'gear': 3, 'key': 'W',
     'held': [[2., 8.], [16., 22.]], 'duration': 30., 'old_name': None},
]


def analyse(directory, case, trace):
    with np.load(directory / 'states.npz', allow_pickle=False) as data:
        pos = data['qpos']
        vel = data['qvel']
        torque = data['applied_torque_nm']
        times = data['segment_time_s']
    with (directory / 'telemetry.csv').open() as stream:
        telemetry = list(csv.DictReader(stream))
    dt = np.diff(times)
    assert len(pos) == len(telemetry)+1 == len(trace)+1
    world_vx = np.diff(pos[:, 0])/dt
    w, x, y, z = pos[:, 3:7].T
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(np.clip(2*(w*y-z*x), -1., 1.))
    direction = 1. if case['key'] == 'W' else -1.
    windows = []
    for release_number, (_, release) in enumerate(case['held'], 1):
        start, end = round(release/.01), round((release+8.)/.01)
        assert end < len(pos), 'all releases need complete 8-second observations'
        progress = direction*(pos[start:end+1, 0]-pos[start, 0])
        increments = np.diff(progress)
        velocities = direction*world_vx[start:end]
        last = world_vx[end-200:end]
        low = np.abs(world_vx[start:end]) < .05
        longest = count = 0
        for item in low:
            count = count+1 if item else 0
            longest = max(longest, count)
        phases = []
        for row in trace[start:end]:
            dynamic = row['dynamic']
            if dynamic and (dynamic['release_started'] or dynamic['hold_entered']):
                phases.append({'time_s': row['time_before_s'], **dynamic})
        window = {
            'release_number': release_number, 'release_time_s': release,
            'observed_seconds': float(times[end]-times[start]),
            'direction': direction, 'release_position_world_m': pos[start, :3].tolist(),
            'net_world_x_m': float(pos[end, 0]-pos[start, 0]),
            'net_along_command_m': float(progress[-1]),
            'peak_forward_excursion_m': float(max(0., np.max(progress))),
            'max_backward_excursion_m': float(max(0., -np.min(progress))),
            'total_backward_travel_m': float(np.maximum(-increments, 0.).sum()),
            'peak_retreat_from_prior_peak_m': float(np.max(np.maximum.accumulate(progress)-progress)),
            'max_reverse_velocity_mps': float(max(0., -np.min(velocities))),
            'last2_mean_signed_world_vx_mps': float(np.mean(last)),
            'last2_mean_abs_world_vx_mps': float(np.mean(np.abs(last))),
            'last2_max_abs_world_vx_mps': float(np.max(np.abs(last))),
            'last2_signed_world_drift_m': float(pos[end, 0]-pos[end-200, 0]),
            'last2_total_world_travel_m': float(np.abs(np.diff(pos[end-200:end+1, 0])).sum()),
            'longest_continuous_abs_world_vx_below_0p05_seconds': longest*.01,
            'last2_all_abs_world_vx_below_0p05': bool(np.all(np.abs(last) < .05)),
            'requested_zero_entire_window': all(row['requested_forward_mps'] == 0. for row in trace[start:end]),
            'accepted_zero_entire_window': all(row['applied_forward_mps'] == 0. for row in trace[start:end]),
            'phase_events': phases,
            'diagnostic_final_speed_below_0p02': bool(np.mean(np.abs(last)) < .02),
            'diagnostic_final_travel_below_0p04': bool(np.abs(np.diff(pos[end-200:end+1, 0])).sum() < .04),
        }
        windows.append(window)
    return {
        'case': case, 'control_transitions': len(torque),
        'duration_s': float(times[-1]), 'termination': 'requested_duration_elapsed',
        'initial_qpos_qvel_sha256': hashlib.sha256(pos[0].tobytes()+vel[0].tobytes()).hexdigest(),
        'finite': bool(np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(torque).all()),
        'max_abs_yaw_drift_rad': float(np.max(np.abs(yaw-yaw[0]))),
        'max_abs_lateral_drift_m': float(np.max(np.abs(pos[:, 1]-pos[0, 1]))),
        'max_abs_roll_pitch_rad': float(max(np.max(np.abs(roll)), np.max(np.abs(pitch)))),
        'recovery_ticks': sum(row['safety_mode'] == 'recovery' for row in telemetry),
        'traction_boost_ticks': sum(row['traction_mode'] == 'boost' for row in telemetry),
        'mean_torque_saturation_fraction': float(np.mean([float(row['torque_saturation_fraction']) for row in telemetry])),
        'reference_release_count': sum(bool(row['dynamic'] and row['dynamic']['release_started']) for row in trace),
        'hold_entry_count': sum(bool(row['dynamic'] and row['dynamic']['hold_entered']) for row in trace),
        'release_windows': windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    from scripts import run_d1_course_drive as runner
    from scripts.d1_course_controls import CourseKeyboardCommands
    from wheel_legged_control.d1.interactive import D1InteractiveSimulation

    work = Path(__file__).resolve().parent
    work_files = [Path(__file__).resolve(), work/'dynamic_brake.py', work/'contract.md']
    work_hashes = {path.name: sha(path) for path in work_files}
    repository_files = sorted((repo/'src').rglob('*.py')) + sorted((repo/'scripts').glob('*.py'))
    repository_hashes = {str(path.relative_to(repo)): sha(path) for path in repository_files}
    protocol = {
        'schema': 'd1-work-dynamic-brake-paired-v1', 'cases': CASES,
        'strategies': ['legacy', 'dynamic_brake_hold'], 'work_source_sha256': work_hashes,
        'repository_source_sha256_before': repository_hashes,
        'constants': {'a_mps2': .5, 'low_speed_mps': .03, 'low_speed_ticks': 30,
                      'control_dt_s': .01, 'release_observation_s': 8.},
        'algorithm': 'See immutable contract.md; symmetric velocity-square cap during brake, then one actual-stop hold anchor.',
        'implementation': 'WorkDrive subclasses the actual course adapter in-process. DynamicBridge supplies only its reference helper. The unchanged GUI runner does the real step and records all states. A writer hook makes every new protocol identify the actual work override before writing it; no record is rewritten.',
        'precompute_timing': 'memory ref/distance precede legacy distance+=body_vx*.01; actual post error is separately recorded.',
        'case_selection': 'Six cases fixed before any run; no scanning or terrain-dependent algorithm.',
        'product_default': 'legacy', 'new_gui_windows': 0,
    }
    write_json(output/'protocol.json', protocol)
    original_drive, original_json = runner.CourseSideStepDrive, runner._json
    results, comparisons = [], []
    try:
        for case in CASES:
            pair = {}
            for strategy in ('legacy', 'dynamic_brake_hold'):
                trace = []
                simulation = D1InteractiveSimulation(arena=case['arena'], baseline='lqr', state_mode='oracle')

                class Bridge:
                    def __init__(self, sim):
                        self.sim, self.core, self.last = sim, DynamicBrakeHold(), None

                    def reset(self):
                        self.core.reset()
                        self.last = None

                    def prepare(self, controller, requested, *, eligible):
                        self.last = self.core.prepare(controller, requested,
                            float(self.sim.teleop._state.base_linear_velocity_body[0]), eligible=eligible)
                        return {'brake_reference_reanchored': self.last['reference_written'],
                                'brake_reference_previous_error_m': self.last['old_error_m'],
                                'brake_reference_new_error_m': self.last['new_error_m'],
                                'brake_reference_event_count': self.last['release_count']}

                    def observe_applied(self, actual, *, eligible):
                        self.core.observe_applied(actual, eligible=eligible)

                class WorkDrive(original_drive):
                    def __init__(self, sim, selected_strategy=strategy, record_sink=trace, **kwargs):
                        super().__init__(sim, **kwargs)
                        self.selected_strategy, self.record_sink = selected_strategy, record_sink
                        if self.selected_strategy == 'dynamic_brake_hold':
                            self.braking = Bridge(sim)

                    def step(self, requested, direction):
                        memory = self.teleop.controller.control_memory
                        row = {'time_before_s': float(self.plant.data.time),
                               'requested_forward_mps': float(requested.forward_velocity_mps),
                               'pre_body_vx_mps': float(self.teleop._state.base_linear_velocity_body[0]),
                               'pre_distance_m': memory.distance_m,
                               'pre_reference_m': memory.distance_reference_m}
                        status, torque, info = super().step(requested, direction)
                        post = self.teleop.controller.control_memory
                        row.update(applied_forward_mps=status.forward_velocity_mps,
                                   post_distance_m=post.distance_m,
                                   post_reference_m=post.distance_reference_m,
                                   post_error_m=post.distance_reference_m-post.distance_m,
                                   safety_mode=status.safety_mode,
                                   dynamic=None if self.selected_strategy == 'legacy' else dict(self.braking.last),
                                   phase_after_observe=None if self.selected_strategy == 'legacy' else self.braking.core.phase)
                        self.record_sink.append(row)
                        if self.selected_strategy != 'legacy':
                            info.update(brake_reference_mode='work_dynamic_brake_hold',
                                        braking_profile='work_dynamic_brake_hold_v1')
                        return status, torque, info

                class Commands(CourseKeyboardCommands):
                    def __init__(self, sim, selected_case=case):
                        self.sim, self.case = sim, selected_case
                        super().__init__(clock=lambda: float(sim.plant.data.time))
                        self.events, self.previous_keys = [], None

                    def __call__(self, seconds):
                        tick = round(seconds/.01)
                        keys = set()
                        if self.case['gear'] >= 2 and tick == 100:
                            keys.add(340)
                        if self.case['gear'] >= 3 and tick == 102:
                            keys.add(344)
                        if any(round(start/.01) <= tick < round(end/.01) for start, end in self.case['held']):
                            keys.add(ord(self.case['key']))
                        if keys != self.previous_keys:
                            self.events.append({'time_s': tick*.01, 'held_keys': sorted(keys)})
                            self.previous_keys = keys.copy()
                        self.update_pressed(keys)
                        return super().__call__(seconds)

                def json_with_actual_implementation(path, data, selected_case=case, selected_strategy=strategy):
                    if Path(path).name == 'protocol.json':
                        data = dict(data)
                        data.update(arena=selected_case['arena'], work_source_sha256=work_hashes,
                                    work_strategy=selected_strategy,
                                    work_override=protocol['implementation'])
                        if selected_strategy != 'legacy':
                            data.update(course_profile='heading_wheel_speed_work_dynamic_brake_v1',
                                        brake_reference_mode='work_dynamic_brake_hold',
                                        braking_profile='work_dynamic_brake_hold_v1',
                                        braking_reference=protocol['algorithm'])
                    original_json(path, data)

                runner.CourseSideStepDrive, runner._json = WorkDrive, json_with_actual_implementation
                directory = output/case['name']/strategy
                directory.parent.mkdir(parents=True, exist_ok=True)
                commands = Commands(simulation)
                runner.run(simulation, directory, seconds=case['duration'], zone=case['zone'],
                           commands=commands, viewer_factory=None, realtime=False,
                           side_step_profile='fast', brake_reference_mode='legacy' if strategy == 'legacy' else 'release_reanchor_experimental')
                with (directory/'control_memory_trace.jsonl').open('x') as stream:
                    for row in trace:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                write_json(directory/'injected_key_events.json', commands.events)
                with tarfile.open(directory/'work_source.tar.gz', 'x:gz') as archive:
                    for path in work_files:
                        archive.add(path, arcname=path.name, recursive=False)
                result = analyse(directory, case, trace)
                result['strategy'] = strategy
                result['torques_within_limits'] = bool(np.all(np.abs(np.load(directory/'states.npz')['applied_torque_nm']) <= simulation.plant.actuator_torque_limit_nm+1e-10))
                write_json(directory/'analysis.json', result)
                pair[strategy] = result
                results.append(result)
                print(json.dumps({'case': case['name'], 'strategy': strategy,
                    'steps': result['control_transitions'], 'windows': [{k: row[k] for k in
                        ('release_number', 'peak_forward_excursion_m', 'peak_retreat_from_prior_peak_m',
                         'max_reverse_velocity_mps', 'last2_mean_abs_world_vx_mps')}
                        for row in result['release_windows']]}), flush=True)
            first = round(case['held'][0][1]/.01)
            comparison = {'case': case['name'], 'first_release_prefix_bitwise': {}, 'release_comparisons': []}
            with np.load(output/case['name']/'legacy/states.npz') as old, np.load(output/case['name']/'dynamic_brake_hold/states.npz') as new:
                for key in ('qpos', 'qvel', 'segment_time_s', 'applied_torque_nm'):
                    stop = first if key == 'applied_torque_nm' else first+1
                    comparison['first_release_prefix_bitwise'][key] = bool(np.array_equal(old[key][:stop], new[key][:stop]))
                if case['old_name']:
                    with np.load(args.historical/'forward_02'/case['old_name']/'states.npz') as historical:
                        comparison['historical_legacy_full_prefix_bitwise'] = {key: bool(np.array_equal(old[key][:len(historical[key])], historical[key])) for key in ('qpos', 'qvel', 'segment_time_s', 'applied_torque_nm')}
            assert all(comparison['first_release_prefix_bitwise'].values()), 'drive prefix changed'
            for old, new in zip(pair['legacy']['release_windows'], pair['dynamic_brake_hold']['release_windows'], strict=True):
                comparison['release_comparisons'].append({
                    'release_number': old['release_number'],
                    'peak_forward_no_worse': new['peak_forward_excursion_m'] <= old['peak_forward_excursion_m']+1e-6,
                    'backward_retreat_no_worse': new['peak_retreat_from_prior_peak_m'] <= old['peak_retreat_from_prior_peak_m']+1e-6,
                    'final_speed_pass': new['diagnostic_final_speed_below_0p02'],
                    'final_travel_pass': new['diagnostic_final_travel_below_0p04'],
                })
            comparisons.append(comparison)
    finally:
        runner.CourseSideStepDrive, runner._json = original_drive, original_json
    current_hashes = {name: sha(repo/name) for name in repository_hashes}
    assert current_hashes == repository_hashes, 'repository sources changed during physics'
    assert {path.name: sha(path) for path in work_files} == work_hashes, 'work implementation changed'
    write_json(output/'summary.json', {'results': results, 'comparisons': comparisons,
        'actual_runs': len(results), 'actual_control_transitions': sum(row['control_transitions'] for row in results),
        'repository_source_unchanged': True, 'work_source_unchanged': True,
        'scope': 'all fixed development cases; legacy remains product default; no parameter search'})
    write_json(output/'manifest.json', {str(path.relative_to(output)): {'sha256': sha(path), 'bytes': path.stat().st_size}
        for path in sorted(output.rglob('*')) if path.is_file()})


if __name__ == '__main__':
    main()
