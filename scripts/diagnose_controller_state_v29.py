"""Describe recorded controller state in complete continuous-task holdout pairs.

Stdlib only; streams existing CSVs, never imports robot/model/physics code.
This is a descriptive follow-up to v28, not a causal test, actor-state
reconstruction, new qualification, or a diagnosis of the separate B22 policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

HELPER = Path(__file__).with_name('compare_continuous_results_v28.py')
_spec = importlib.util.spec_from_file_location('continuous_pairs_v28', HELPER)
pairs = importlib.util.module_from_spec(_spec)
_previous_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    _spec.loader.exec_module(pairs)
finally:
    sys.dont_write_bytecode = _previous_bytecode

SCHEMA = 'continuous-controller-state-diagnostic-v29'
DT = .01
TOL = 1e-8
ENTRY_S = 1.
SIGN_EPS = 1e-10
REQUIRED = {'step', 'time_s', 'stage', 'stage_time_s', 'terminated', 'truncated', 'termination_reason'}
RECORDED = ('yaw_integral_nm', 'command_yaw_rate_rps', 'yaw_differential_torque_nm',
            'yaw_rate_error_rps', 'clearance_error_m', 'residual_longitudinal_n',
            'residual_vertical_n', 'action_longitudinal', 'action_vertical',
            'torque_saturation_fraction')
MISSING_STATE = ('lqr_distance_m', 'lqr_distance_reference_m', 'actor_observation_vector',
                 'baseline_longitudinal_force_n', 'baseline_support_force_n')


def finite(value):
    result = float(value)
    pairs.require(math.isfinite(result), 'nonfinite diagnostic telemetry')
    return result


class Window:
    def __init__(self, available):
        self.available = available
        self.n = self.eligible = self.opposed = self.saturated = self.vertical_clip = 0
        self.sums = dict.fromkeys(available, 0.)
        self.squares = dict.fromkeys(available, 0.)
        self.first = self.last = None
        self.maximum_integral = 0.

    def add(self, row):
        self.n += 1
        values = {key: finite(row[key]) for key in self.available}
        for key, value in values.items():
            self.sums[key] += value
            self.squares[key] += value*value
        integral = values.get('yaw_integral_nm')
        if integral is not None:
            if self.first is None:
                self.first = integral
            self.last = integral
            self.maximum_integral = max(self.maximum_integral, abs(integral))
        command = values.get('command_yaw_rate_rps')
        if integral is not None and command is not None and abs(command) > SIGN_EPS:
            self.eligible += 1
            self.opposed += (integral > SIGN_EPS and command < 0) or (integral < -SIGN_EPS and command > 0)
        if 'torque_saturation_fraction' in values:
            fraction = values['torque_saturation_fraction']
            pairs.require(0 <= fraction <= 1, 'invalid recorded torque saturation fraction')
            self.saturated += fraction > 0
        for key in ('action_longitudinal', 'action_vertical'):
            if key in values:
                pairs.require(abs(values[key]) <= 1+1e-7, 'recorded applied action outside [-1,1]')
        if 'action_vertical' in values:
            self.vertical_clip += abs(values['action_vertical']) >= 1-1e-7

    def finish(self):
        def mean(key):
            return self.sums[key]/self.n if key in self.sums and self.n else None

        def rms(key):
            return math.sqrt(self.squares[key]/self.n) if key in self.squares and self.n else None

        def count_ratio(numerator, available):
            return {'ticks': numerator, 'denominator_ticks': self.n,
                    'fraction': numerator/self.n if self.n else None} if available else None

        integral = 'yaw_integral_nm' in self.available
        opposite = integral and 'command_yaw_rate_rps' in self.available
        return {
            'ticks': self.n, 'recorded_seconds': self.n*DT,
            'yaw_integral_nm': {'first': self.first, 'last': self.last,
                                'mean': mean('yaw_integral_nm'),
                                'max_abs': self.maximum_integral if self.n else None} if integral else None,
            'integral_opposes_command': {
                'ticks': self.opposed, 'denominator_commanded_yaw_ticks': self.eligible,
                'fraction': self.opposed/self.eligible if self.eligible else None} if opposite else None,
            'yaw_rate_error_rmse_rps': rms('yaw_rate_error_rps'),
            'clearance_error_rmse_m': rms('clearance_error_m'),
            'mean_yaw_differential_torque_nm': mean('yaw_differential_torque_nm'),
            'mean_residual_longitudinal_n': mean('residual_longitudinal_n'),
            'mean_residual_vertical_n': mean('residual_vertical_n'),
            'mean_torque_saturation_fraction': mean('torque_saturation_fraction'),
            'ticks_with_any_recorded_torque_saturation': count_ratio(
                self.saturated, 'torque_saturation_fraction' in self.available),
            'vertical_action_at_clip': count_ratio(self.vertical_clip, 'action_vertical' in self.available),
        }


def summarize_telemetry(path, duration, stages, force_scales):
    """One pass, bounded memory; incomplete clocks cannot become stage summaries."""
    duration = finite(duration)
    pairs.require(duration > 0 and abs(round(duration/DT)*DT-duration) <= TOL,
                  'duration is not a positive complete control horizon')
    pairs.require(stages and stages[0][1] == 0 and stages[-1][2] == 45., 'unsupported stage template')
    names = [stage[0] for stage in stages]
    pairs.require(len(names) == len(set(names)), 'duplicate stage labels')
    for index, stage in enumerate(stages):
        pairs.require(len(stage) == 5 and stage[1] < stage[2]
                      and (index == 0 or stage[1] == stages[index-1][2]), 'invalid stage template')
    windows, digest, byte_count, count, ended = {}, hashlib.sha256(), 0, 0, False
    with Path(path).open('rb') as stream:
        def lines():
            nonlocal byte_count
            for line in stream:
                byte_count += len(line)
                digest.update(line)
                yield line.decode('utf-8')

        reader = csv.DictReader(lines())
        header = reader.fieldnames
        pairs.require(header and len(header) == len(set(header)) and REQUIRED <= set(header),
                      'missing or duplicate structural CSV columns')
        available = tuple(key for key in RECORDED if key in header)
        windows = {(name, window): Window(available) for name in names
                   for window in ('entry_first_1s', 'after_1s')}
        for row in reader:
            pairs.require(None not in row and None not in row.values(), 'malformed CSV row')
            pairs.require(not ended, 'telemetry continues after completion')
            count += 1
            pairs.require(pairs.integer(row['step'], 'step') == count
                          and abs(finite(row['time_s'])-count*DT) <= TOL,
                          'missing, duplicate or shifted control tick')
            template_time = (count-1)*DT*45./duration
            plan = next((stage for stage in stages if template_time < stage[2]), stages[-1])
            elapsed = (template_time-plan[1])*duration/45.+DT
            pairs.require(row['stage'] == plan[0] and abs(finite(row['stage_time_s'])-elapsed) <= TOL,
                          'stage or stage clock disagrees with bound protocol')
            pairs.require(row['terminated'] == '0' and row['truncated'] in ('0', '1'),
                          'terminated or malformed completion flags')
            ended = row['truncated'] == '1'
            pairs.require(row['termination_reason'] == ('time_limit' if ended else 'ongoing'),
                          'completion reason disagrees with flags')
            for action, force, scale in zip(('action_longitudinal', 'action_vertical'),
                                           ('residual_longitudinal_n', 'residual_vertical_n'),
                                           force_scales, strict=True):
                if action in available and force in available:
                    pairs.require(abs(finite(row[action])*scale-finite(row[force])) <= 1e-6,
                                  'recorded residual force disagrees with action scale')
            window = 'entry_first_1s' if elapsed <= ENTRY_S+TOL else 'after_1s'
            windows[plan[0], window].add(row)
    pairs.require(ended and count == round(duration/DT), 'telemetry is not a complete requested horizon')
    inventory = {key: 'RECORDED' if key in header else 'NOT_RECORDED' for key in (*RECORDED, *MISSING_STATE)}
    inventory['route_reference_x_y_yaw'] = (
        'RECORDED_ROUTE_REFERENCE_NOT_LQR_MEMORY'
        if {'reference_x_m', 'reference_y_m', 'reference_yaw_rad'} <= set(header) else 'NOT_RECORDED')
    return {'ticks': count, 'field_inventory': inventory,
            'stage_windows': [{'stage': stage, 'window': window, **values.finish()}
                              for (stage, window), values in windows.items()]}, {
                'path': str(Path(path).resolve()), 'bytes': byte_count, 'sha256': digest.hexdigest()}


def descriptive_index(cases):
    """Keep training seeds separate and retain actual tick denominators."""
    result = []
    for seed in sorted({case['training_seed'] for case in cases}):
        selected = [case for case in cases if case['training_seed'] == seed]
        windows = [window for case in selected for window in case['stage_windows']]
        right = [window for window in windows if window['stage'] == 'turn_right'
                 and window['window'] == 'entry_first_1s']
        row = {'training_seed': seed, 'mode': selected[0]['mode'], 'cases': len(selected),
               'control_ticks': sum(window['ticks'] for window in windows)}
        for key in ('vertical_action_at_clip', 'ticks_with_any_recorded_torque_saturation'):
            row[key] = ({'ticks': sum(window[key]['ticks'] for window in windows),
                         'denominator_ticks': row['control_ticks']}
                        if all(window[key] is not None for window in windows) else None)
        row['turn_right_entry'] = None
        if right and all(window['yaw_integral_nm'] is not None
                         and window['integral_opposes_command'] is not None for window in right):
            row['turn_right_entry'] = {
                'cases': len(right), 'ticks': sum(window['ticks'] for window in right),
                'first_recorded_integral_nm_range': [min(window['yaw_integral_nm']['first'] for window in right),
                                                     max(window['yaw_integral_nm']['first'] for window in right)],
                'opposed_integral_ticks': sum(window['integral_opposes_command']['ticks'] for window in right),
                'denominator_commanded_yaw_ticks': sum(
                    window['integral_opposes_command']['denominator_commanded_yaw_ticks'] for window in right),
            }
        result.append(row)
    return result


def diagnose(run_dir, audit_dir):
    run_dir, audit_dir = Path(run_dir).resolve(), Path(audit_dir).resolve()
    protocol, metrics, bindings, paths = pairs.load_inputs(run_dir, audit_dir)
    pairing = pairs.compare_rows(protocol, metrics)
    selected = [pair for pair in pairing['pairs'] if pair['split'] == 'holdout']
    invalid = [pair for pair in selected if pair['status'] != 'COMPARABLE']
    result = {'schema': SCHEMA, 'status': 'INCOMPARABLE' if invalid else 'DIAGNOSED',
              'scope': 'continuous-task final holdout paired logs; not B22 current actor',
              'holdout_pair_count': len(selected), 'invalid_pairs': invalid,
              'input_identities': bindings, 'raw_inputs': {}, 'cases': [],
              'entry_sha256': pairs.sha(Path(__file__).read_bytes()),
              'pairing_helper_sha256': pairs.sha(HELPER.read_bytes()),
              'window_definition': 'first 1 s of each stage versus remaining ticks; descriptive, not a qualification gate',
              'sign_epsilon': SIGN_EPS, 'action_clip_count_tolerance': 1e-7,
              'replication_unit': 'training seed; repeated baseline cases are not independent replications',
              'limitations': ['recorded state summaries are not evidence of causal influence',
                             'no full actor input or LQR distance memory reconstruction',
                             'post-step truth yaw error is not the PI pre-step sensor error',
                             'first integral sample is after that tick PI update, not pre-entry memory',
                             'not being explicit in the actor input does not prove unobservability',
                             'no raw NPZ/model/source archive re-audit or new quality qualification'],
              'models_loaded': 0, 'gui_starts': 0, 'physics_steps': 0}
    if invalid:
        return result
    pairs.require(protocol['observation_schema'] == 'd1-continuous-sensor-command45-v1'
                  and protocol['control_schema'] == 'd1-lqr-vmc-sensor-world-slopes-yaw-pi-v2'
                  and protocol['action_schema'] == 'd1-terrain-residual-quarter-v1'
                  and protocol['residual_force_scale_n'] == [11.25, 20.], 'unsupported control/observation/force contract')
    manifest = pairs.strict_json(paths['run_manifest'].read_bytes())['sha256']
    plans = pairs.expected_matrix(protocol)
    names = sorted({name for pair in selected for name in (pair['rollout'], 'zero_'+pair['case'])})
    for name in names:
        plan = plans[name]
        relative = f'rollouts/{name}/protocol.json'
        data = (run_dir/relative).read_bytes()
        pairs.require(pairs.sha(data) == manifest[relative], 'rollout protocol SHA mismatch')
        config = pairs.strict_json(data)
        metadata = config['checkpoint_metadata'] or {}
        pairs.require(config['mode'] == plan['mode']
                      and f"seed{config['seed']}_{config['config']['duration_s']:g}s" == plan['case']
                      and config['control_dt_s'] == DT and config['config']['yaw_controller'] == 'pi'
                      and metadata.get('training_seed', 0) == plan['training_seed']
                      and metadata.get('num_timesteps', 0) == plan['budget']
                      and all(config[key] == protocol[key] for key in
                              ('control_schema', 'observation_schema', 'action_schema')),
                      'rollout metadata disagrees with paired protocol')
        result['raw_inputs'][relative] = {'path': str(run_dir/relative), 'bytes': len(data), 'sha256': pairs.sha(data)}
        relative = f'rollouts/{name}/telemetry.csv'
        summary, identity = summarize_telemetry(run_dir/relative, plan['expected_duration_s'],
                                               protocol['stages_45_second_template'],
                                               protocol['residual_force_scale_n'])
        pairs.require(identity['sha256'] == manifest[relative], 'raw telemetry SHA mismatch')
        result['raw_inputs'][relative] = identity
        result['cases'].append({**plan, **summary})
    pairs.require(all(pairs.sha(path.read_bytes()) == bindings[key]['sha256'] for key, path in paths.items()),
                  'small input identity changed during diagnosis')
    result['unique_case_count'] = len(names)
    result['total_control_ticks'] = sum(case['ticks'] for case in result['cases'])
    result['training_seed_groups'] = [{'training_seed': seed,
                                      'holdout_pairs': sum(pair['training_seed'] == seed for pair in selected)}
                                     for seed in protocol['training_seeds']]
    result['descriptive_index'] = descriptive_index(result['cases'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new JSON outside both input directories')
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        pairs.require(not output.exists() and not args.output.is_symlink()
                      and not any(output.is_relative_to(path.resolve()) for path in (args.run, args.audit)),
                      'output must be new and outside run/audit inputs')
        result = diagnose(args.run, args.audit)
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, KeyError, TypeError, OverflowError, csv.Error) as exc:
        print(json.dumps({'status': 'ERROR', 'error': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({'status': result['status'], 'holdout_pairs': result['holdout_pair_count'],
                      'unique_cases': result.get('unique_case_count'),
                      'control_ticks': result.get('total_control_ticks'), 'physics_steps': 0}))
    return 0 if result['status'] == 'DIAGNOSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
