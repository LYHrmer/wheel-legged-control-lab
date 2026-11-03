"""Independently audit frozen MPC82 CSVs using the Python standard library.

This reads saved data only. It never imports the candidate runner, legacy metric
functions, NumPy, MuJoCo or a controller, and never evaluates a dynamics model.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import itertools
import json
import math
import resource
import signal
import struct
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path('/home/lyh/wheel-legged-control-lab')
STUDY = 'results/delay_mpc82_20261103_01'
OLD = 'results/mpc49_preflight_20261022_01'
PLAN = 'docs/experiments82/delay_mpc82_preregistered.json'
PLAN_SHA = 'ea7847eafc8deca1b38879523708b963a3743b74cf04193d5ca916b0343f2b76'
REG49 = 'reports/control_candidates49_preregistration.json'
SEEDS = (490101, 490102, 490103)
SCENARIOS = ('nominal', 'mismatch_delay')
ARMS = ('baseline', 'candidate')
STATE = ('x', 'pitch', 'extension', 'vx', 'pitch_rate', 'extension_rate')
CORE = ('time_s', *STATE, 'wheel_force', 'leg_force', 'target_vx',
        'target_extension', 'push', 'reward')
EXTRA = ('next_solve_ms', 'issued_wheel_force', 'issued_leg_force',
         'pending0_wheel', 'pending0_leg', 'pending1_wheel', 'pending1_leg',
         *[f'measured_{i}' for i in range(6)], *[f'noise_{i}' for i in range(6)],
         'next_iterations')
GUARDS = ('velocity_rmse_mps', 'pitch_rmse_deg', 'max_abs_pitch_deg',
          'normalized_effort_rms', 'wheel_force_rms_n', 'leg_force_rms_n')
CAP = 2 << 20
ATOL, RTOL = 1e-10, 1e-9


def require(ok, message):
    if not ok:
        raise ValueError(message)


def mean(values):
    require(bool(values), 'empty mean')
    return math.fsum(values) / len(values)


def rms(values):
    return math.sqrt(mean([v * v for v in values]))


def percentile95(values):
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def metrics(rows, solves):
    wheel = [r['wheel_force'] for r in rows]
    leg = [r['leg_force'] for r in rows]
    return {
        'height_rmse_mm': 1000 * rms([r['extension'] - r['target_extension'] for r in rows]),
        'velocity_rmse_mps': rms([r['vx'] - r['target_vx'] for r in rows]),
        'pitch_rmse_deg': rms([math.degrees(r['pitch']) for r in rows]),
        'max_abs_pitch_deg': max(abs(math.degrees(r['pitch'])) for r in rows),
        'normalized_effort_rms': rms([v / 80 for v in wheel] + [v / 260 for v in leg]),
        'wheel_force_rms_n': rms(wheel), 'leg_force_rms_n': rms(leg),
        'mean_reward': mean([r['reward'] for r in rows]),
        'solve_p95_ms': percentile95(solves),
        'success': int(all(abs(r['pitch']) < 0.70 for r in rows)),
    }


def height_description(rows, lower):
    errors = [r['extension'] - r['target_extension'] for r in rows]
    bias, mse = mean(errors), mean([e * e for e in errors])
    tail = [r for r in rows if 6.0 < r['time_s'] <= 8.0]
    require(bool(tail), 'last-two-second descriptive segment absent')
    extension = [r['extension'] for r in tail]
    return {
        'rows': len(rows), 'error_bias_m': bias, 'error_mse_m2': mse,
        'error_rmse_mm': 1000 * math.sqrt(mse),
        'bias_squared_fraction_of_mse': bias * bias / mse if mse else None,
        'centered_error_variance_m2': mean([(e - bias)**2 for e in errors]),
        'last_two_seconds': {
            'interval': '6.0 < post-transition time_s <= 8.0; posthoc, not a gate',
            'rows': len(tail), 'extension_mean_m': mean(extension),
            'extension_min_m': min(extension), 'extension_max_m': max(extension),
            'target_extension_mean_m': mean([r['target_extension'] for r in tail]),
            'height_error_mean_m': mean([r['extension'] - r['target_extension'] for r in tail]),
            'xml_lower_limit_m': lower,
            'mean_signed_distance_from_lower_limit_m': mean([v - lower for v in extension]),
            'mean_absolute_distance_from_lower_limit_m': mean([abs(v - lower) for v in extension]),
        },
    }


def npz_headers_and_payload(raw):
    """Decode finite little-endian F64 NPY members without pickle or NumPy."""
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 8, 'unique eight-member model snapshot required')
        require(sum(i.file_size for i in archive.infolist()) < 1 << 20, 'snapshot size cap')
        for name in names:
            data = archive.read(name)
            require(data[:6] == b'\x93NUMPY', 'NPY magic')
            require(data[6:8] in (b'\x01\x00', b'\x02\x00'), 'NPY version')
            offset = 10 if data[6] == 1 else 12
            length = int.from_bytes(data[8:offset], 'little')
            require(length < 65536, 'NPY header cap')
            header = ast.literal_eval(data[offset:offset + length].decode('latin1'))
            require(header['descr'] == '<f8' and header['fortran_order'] is False, 'F64 C-order snapshot')
            shape = header['shape']
            require(isinstance(shape, tuple) and all(type(n) is int and n > 0 for n in shape), 'NPY shape')
            payload = data[offset + length:]
            require(len(payload) == math.prod(shape) * 8, 'NPY payload length')
            values = [v[0] for v in struct.iter_unpack('<d', payload)]
            require(all(math.isfinite(v) for v in values), 'nonfinite snapshot')
            arrays[name[:-4]] = {'shape': shape, 'values': values,
                                'payload_sha256': hashlib.sha256(payload).hexdigest()}
    require(set(arrays) == {'a', 'b', 'q', 'r', 'terminal_q', 'sx', 'su', 'equilibrium'}, 'snapshot keys')
    return arrays


class Reader:
    def __init__(self, report):
        self.report = report

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        bound = self.report['inputs_sha256']
        require(str(path) not in bound or bound[str(path)] == digest, 'input changed while reading')
        bound[str(path)] = digest
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path, new=True):
        source = csv.DictReader(io.StringIO(self.read(path).decode()))
        require(len(source.fieldnames) == len(set(source.fieldnames)), 'duplicate CSV column')
        require(set(CORE) <= set(source.fieldnames), 'missing original CSV columns')
        if new:
            require(set(source.fieldnames) == set(CORE + EXTRA), 'unexpected new CSV schema')
        rows = [{k: float(v) for k, v in row.items()} for row in source]
        require(len(rows) == 400, 'incomplete CSV')
        require(all(math.isfinite(v) for row in rows for v in row.values()), 'nonfinite CSV')
        return rows

    def close(self, a, b, label):
        require(math.isfinite(a) and math.isfinite(b), 'nonfinite comparison: ' + label)
        gap = abs(a - b)
        self.report['scalar_checks'] += 1
        gaps = self.report['maximum_comparison_gaps']
        gaps[label] = max(gaps.get(label, 0.0), gap)
        require(gap <= ATOL + RTOL * abs(b), 'value mismatch: ' + label)


def audit(report, deadline):
    reader = Reader(report)
    root = ROOT / STUDY
    original = reader.json(root / 'report.json')
    plan = reader.json(ROOT / PLAN)
    copied_plan = reader.json(root / 'plan.json')
    require(report['inputs_sha256'][str(ROOT / PLAN)] == PLAN_SHA, 'sealed plan SHA')
    require(report['inputs_sha256'][str(root / 'plan.json')] == PLAN_SHA, 'copied plan SHA')
    require(plan == copied_plan and original['plan_sha256'] == PLAN_SHA, 'plan/report binding')
    expected_settings = {
        'preflight_seeds': list(SEEDS), 'scenarios': list(SCENARIOS), 'profiles': ['benchmark'],
        'arms': list(ARMS), 'episodes': 12, 'episode_control_steps': 400,
        'control_steps_cap': 4800, 'integration_steps_cap': 48000, 'expected_solve_calls': 4812,
        'control_dt_s': 0.02, 'integration_dt_s': 0.002, 'horizon': 20, 'q_height': 90.0,
        'candidate_delay_steps': {'nominal': 0, 'mismatch_delay': 2},
        'training_steps': 0, 'full_phase_authorized': False,
    }
    require(all(plan[k] == value for k, value in expected_settings.items()), 'registered settings')
    require(original['status'] == original['verification_after'] == 'COMPLETE', 'original execution incomplete')
    require(original['failure'] is None and original['full_status'] == 'NOT_RUN', 'original failure/full status')
    require(original['training_steps'] == original['archive48_invocations'] == 0
            and original['runtime_v27_changed'] is False, 'original scope changed')
    require(original['source_sha256'] == plan['source_sha256'] and len(plan['source_sha256']) == 32,
            '32-source closure')
    old_registration = reader.json(ROOT / REG49)
    for name, digest in old_registration['source_sha256'].items():
        require(plan['source_sha256'][name] == digest, 'original49 frozen source binding')
    for name, digest in plan['source_sha256'].items():
        deadline()
        reader.read(ROOT / name)
        require(report['inputs_sha256'][str(ROOT / name)] == digest, 'source SHA: ' + name)
    require(original['versions'] == plan['versions'] == reader.json(ROOT / OLD / 'experiment.json')['versions'],
            'recorded numerical versions differ')
    artifacts = original['raw_artifacts_sha256']
    require(len(artifacts) == 50, 'fifty original artifact bindings required')
    actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual_files == set(artifacts) | {'report.json'}, 'unexpected or missing original artifact')
    total_bytes = 0
    for name, digest in artifacts.items():
        deadline()
        data = reader.read(root / name)
        total_bytes += len(data)
        require(report['inputs_sha256'][str(root / name)] == digest, 'artifact SHA: ' + name)
    require(total_bytes == original['raw_bytes_before_report'], 'original artifact byte count')
    report['binding_before'] = {'status': 'COMPLETE', 'source_count': 32, 'raw_artifact_count': 50,
                                'raw_bytes': total_bytes, 'recorded_versions': original['versions']}
    xml = ET.fromstring(reader.read(ROOT / 'src/wheel_legged_control/assets/wheel_legged_planar.xml'))
    extension = xml.find(".//joint[@name='leg_extension']")
    lower, upper = map(float, extension.attrib['range'].split())
    require((lower, upper) == (-0.12, 0.12), 'frozen extension range')
    grid = {(seed, scenario, arm) for seed, scenario, arm in itertools.product(SEEDS, SCENARIOS, ARMS)}
    episodes = original['episodes']
    identities = [(e['seed'], e['scenario'], e['arm']) for e in episodes]
    require(len(identities) == 12 and set(identities) == grid, 'complete unique episode grid')
    traces, matrices, reconstructed = {}, {}, []
    for episode in episodes:
        deadline()
        key = (episode['seed'], episode['scenario'], episode['arm'])
        folder = f'{key[0]}_{key[1]}_benchmark_{key[2]}'
        saved = reader.json(root / folder / 'metrics.json')
        require(saved == episode, 'episode receipt/report identity')
        require(episode['profile'] == 'benchmark' and episode['completed'] is True, 'incomplete episode')
        counts = {'steps': 400, 'control_attempted': 400, 'physics_steps': 4000,
                  'queue_checks_completed': 400, 'solve_calls': 401, 'initial_solve_calls': 1,
                  'control_decisions_applied': 400, 'post_transition_compute_calls': 400,
                  'terminal_unapplied_compute_calls': 1}
        require(all(episode[k] == v for k, v in counts.items()), 'per-episode accounting')
        require(episode['nominal_model_preserved'] is True, 'original final model check failed')
        reader.close(episode['last_simulation_time_s'], 8.0, 'native_simulation_time')
        require(round(episode['last_simulation_time_s'] / 0.002) == 4000, 'native-step reconstruction')
        rows = reader.csv(root / folder / 'trajectory.csv')
        require(report['inputs_sha256'][str(root / folder / 'trajectory.csv')] == episode['trace_sha256'],
                'episode CSV SHA')
        arrays = npz_headers_and_payload(reader.read(root / folder / 'nominal_prediction_model.npz'))
        delay = 0 if key[1] == 'nominal' else 2
        require(episode['delay_steps'] == delay, 'physical delay by scenario')
        expected_shapes = {'a': (6, 6), 'b': (6, 2), 'q': (6, 6), 'r': (2, 2),
                           'terminal_q': (6, 6), 'equilibrium': (2,), 'su': (120, 40),
                           'sx': (120, 10 if delay and key[2] == 'candidate' else 6)}
        require(all(arrays[k]['shape'] == shape for k, shape in expected_shapes.items()), 'snapshot shapes')
        require([arrays['q']['values'][i * 6 + i] for i in range(6)] == [0.4, 140, 90, 4, 14, 5],
                'Q unchanged')
        require(arrays['r']['values'] == [0.025, 0.0, 0.0, 0.004], 'R unchanged')
        queue = [list(arrays['equilibrium']['values']) for _ in range(delay)]
        require(queue == episode['initial_pending_controls'], 'initial equilibrium FIFO')
        solves = episode['solve_samples_ms']
        require(len(solves) == 401 and all(math.isfinite(v) and v >= 0 for v in solves), 'finite 401 solve samples')
        record = {'seed': key[0], 'scenario': key[1], 'arm': key[2], 'status': 'INCOMPLETE',
                  'rows_checked': 0, 'fifo_checks': 0}
        report['episodes'].append(record)
        for index, row in enumerate(rows):
            if index % 64 == 0:
                deadline()
            reader.close(row['time_s'], (index + 1) * 0.02, 'post_transition_timestamps')
            require(row['next_solve_ms'] == solves[index + 1], 'CSV/receipt solve ordering')
            require(row['next_iterations'] >= 0 and row['next_iterations'].is_integer(), 'iteration field')
            pending = [[row[f'pending{i}_wheel'], row[f'pending{i}_leg']] for i in range(2)]
            require(pending == (queue if delay else [[0.0, 0.0], [0.0, 0.0]]), 'logged FIFO before')
            issued = [row['issued_wheel_force'], row['issued_leg_force']]
            applied = [row['wheel_force'], row['leg_force']]
            require(all(-80 <= force[0] <= 80 and 0 <= force[1] <= 260 for force in (issued, applied)),
                    'original force bounds')
            queue.append(issued)
            require(queue.pop(0) == applied, 'actual append/pop delayed output')
            previous = episode['initial_state'] if index == 0 else [rows[index - 1][k] for k in STATE]
            for coordinate in range(6):
                require(row[f'measured_{coordinate}'] - previous[coordinate] == row[f'noise_{coordinate}'],
                        'pretransition measurement/noise alignment')
                if delay == 0:
                    require(row[f'noise_{coordinate}'] == 0, 'nominal zero-noise condition')
            require(abs(row['pitch']) < 0.70, 'original fallen termination threshold')
            reward = (0.45 * math.exp(-18 * row['pitch']**2 - 0.35 * row['pitch_rate']**2)
                      + 0.35 * math.exp(-1.8 * (row['vx'] - row['target_vx'])**2)
                      + 0.20 * math.exp(-90 * (row['extension'] - row['target_extension'])**2
                                        - 0.08 * row['extension_rate']**2))
            reader.close(reward, row['reward'], 'zero_residual_reward')
            record['rows_checked'] += 1
            record['fifo_checks'] += 1
            report['rows_checked'] += 1
        computed = metrics(rows, solves)
        for name, value in computed.items():
            reader.close(value, episode[name], 'metric_' + name)
        require(episode['metric_crosscheck']['passed'] is True, 'original metric crosscheck failed')
        record.update(status='COMPLETE', metrics=computed, height_posthoc=height_description(rows, lower),
                      final_logged_FIFO=queue,
                      snapshot_payload_sha256={name: a['payload_sha256'] for name, a in arrays.items()},
                      original_final_snapshot_check=episode['nominal_model_preserved'])
        traces[key], matrices[key] = rows, arrays
        reconstructed.append(dict(seed=key[0], scenario=key[1], arm=key[2], **computed))
    require(report['rows_checked'] == 4800, 'complete row count')
    baseline_replays, nominal_pairs = [], []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            rows = traces[(seed, scenario, 'baseline')]
            old_rows = reader.csv(ROOT / OLD / f'{seed}_{scenario}_benchmark_baseline/trajectory.csv', new=False)
            require(all(a[k] == b[k] for a, b in zip(rows, old_rows, strict=True) for k in CORE),
                    'original49 baseline core identity')
            baseline_replays.append({'seed': seed, 'scenario': scenario, 'rows': 400, 'exact': True})
        baseline, candidate = (traces[(seed, 'nominal', arm)] for arm in ARMS)
        require(all(a[k] == b[k] for a, b in zip(baseline, candidate, strict=True) for k in CORE),
                'nominal candidate/baseline core identity')
        require(all(matrices[(seed, 'nominal', 'baseline')][name]['payload_sha256']
                    == matrices[(seed, 'nominal', 'candidate')][name]['payload_sha256']
                    for name in matrices[(seed, 'nominal', 'baseline')]), 'nominal condensation identity')
        nominal_pairs.append({'seed': seed, 'rows': 400, 'exact': True})
    first_arrays = next(iter(matrices.values()))
    physical = ('a', 'b', 'q', 'r', 'terminal_q', 'equilibrium')
    require(all(a[name]['payload_sha256'] == first_arrays[name]['payload_sha256']
                for a in matrices.values() for name in physical), 'shared initial model/cost differs')
    report['independent_integrity'] = {
        'all_metric_crosschecks': True, 'six_original49_baselines_exact': True,
        'all_three_nominal_candidate_baseline_exact': True, 'all_queue_checks': True,
        'all_original_model_snapshots_preserved': True,
    }
    require(report['independent_integrity'] == original['decision']['integrity_gates'], 'integrity gate record')
    report['baseline49_exact'] = baseline_replays
    report['nominal_exact'] = nominal_pairs
    report['identity_columns'] = list(CORE)
    report['identity_excludes'] = ['solver timing; iteration diagnostics are not original core columns']

    def ratio(scenario, metric):
        values = {arm: mean([e[metric] for e in reconstructed if e['scenario'] == scenario and e['arm'] == arm])
                  for arm in ARMS}
        require(values['baseline'] > 0, 'zero metric denominator in this saved cohort')
        return values['candidate'] / values['baseline']

    details = {'mismatch_height_ratio': ratio('mismatch_delay', 'height_rmse_mm')}
    gates = {'height_improvement_at_least_10_percent': details['mismatch_height_ratio'] <= 0.90}
    for scenario in SCENARIOS:
        for metric in GUARDS:
            value = ratio(scenario, metric)
            details[f'{scenario}/{metric}_ratio'] = value
            gates[f'{scenario}/{metric}_within_5_percent'] = value <= 1.05
    p95 = max(e['solve_p95_ms'] for e in reconstructed if e['arm'] == 'candidate')
    details['candidate_max_episode_solve_p95_ms'] = p95
    gates['solve_p95_within_20_ms'] = p95 <= 20
    require(gates == original['decision']['performance_gates'], 'original performance gate booleans')
    require(set(details) == set(original['decision']['details']), 'original detail keys')
    for name, value in details.items():
        reader.close(value, original['decision']['details'][name], 'decision_' + name)
    accepted = all(gates.values()) and all(report['independent_integrity'].values())
    require(accepted == original['decision']['accepted'], 'accepted decision mismatch')
    require(original['decision']['all_episodes_complete'] is True
            and original['decision']['cohort_valid'] is True, 'original cohort flags')
    report['recomputed_decision'] = {
        'accepted': accepted, 'performance_gates': gates, 'details': details,
        'failed_performance_gates': [k for k, passed in gates.items() if not passed],
        'full_status': 'NOT_RUN', 'qualification': 'original development preflight only',
    }
    report['height_posthoc_pooled'] = [
        dict(scenario=scenario, arm=arm, **height_description(
            [r for seed in SEEDS for r in traces[(seed, scenario, arm)]], lower))
        for scenario, arm in itertools.product(SCENARIOS, ARMS)
    ]
    counts = {'control_steps': report['rows_checked'], 'integration_steps': 10 * report['rows_checked'],
              'solve_calls': sum(len(e['solve_samples_ms']) for e in episodes),
              'initial_solve_calls': len(episodes)}
    require(all(original[k] == v for k, v in counts.items()), 'whole-cohort accounting')
    require(tuple(counts.values()) == (4800, 48000, 4812, 12), 'exact whole-cohort totals')
    report['original_execution_counts_reconstructed'] = counts
    report['original_execution_resource_receipt'] = {
        'cpu_seconds': original['cpu_seconds'], 'wall_seconds': original['wall_seconds'],
        'source': 'saved original runner receipt; not independently measured by this audit',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'output exists; no overwrite/resume')
    initial = args.output.with_name(args.output.name + '.initial.json')
    script = Path(__file__).resolve()
    report = {
        'schema': 'mpc-independent82-readback-v1', 'status': 'INCOMPLETE',
        'verification_after': 'NOT_RUN', 'new_physics_steps': 0, 'new_model_calls': 0,
        'new_parameter_updates': 0, 'rows_checked': 0, 'scalar_checks': 0,
        'inputs_sha256': {str(script): hashlib.sha256(script.read_bytes()).hexdigest()},
        'maximum_comparison_gaps': {}, 'episodes': [],
        'tolerance': {'absolute': ATOL, 'relative': RTOL,
                      'exact_checks': 'hashes, recorded FIFO/noise, nominal core and old49 core'},
        'interpretation_limits': [
            'Audit COMPLETE means arithmetic/integrity checks passed, not that the controller passed.',
            'Three already-observed development noise seeds; nominal repeats are not independent stochastic evidence.',
            'Combined mass/damping/noise/delay/push stress does not isolate the physical cause of degradation.',
            'Height bias and final-two-second proximity to a soft XML limit are posthoc descriptions, not proof of an offset cause.',
            'All physical prediction matrices are stored initial snapshots. Their reported unchanged final values are execution assertions, not independently stored final arrays.',
            'Recorded MuJoCo time plus frozen integration schedule support native-step counts; this audit did not observe the original mj_step calls.',
            '401 solver-time samples include reset and the final unapplied computation; timing is descriptive, not a solver optimality certificate.',
            'Force bounds are actuator limits, not a proof of deployment or hardware safety.',
            'No frozen gate is changed and no full comparison, training or simulation is run.',
        ],
    }
    with initial.open('x') as stream:
        stream.write(json.dumps({'status': 'INCOMPLETE', 'schema': report['schema'],
                                 'script_sha256': report['inputs_sha256'][str(script)]}) + '\n')
    cpu_start, wall_start = time.process_time(), time.monotonic()

    def stop(signum, _frame):
        raise InterruptedError('audit interrupted: signal ' + str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU, signal.SIGALRM):
        signal.signal(sig, stop)
    resource.setrlimit(resource.RLIMIT_CPU, (29, 30))
    signal.alarm(60)

    def deadline():
        require(time.process_time() - cpu_start < 25, 'cooperative CPU budget')
        require(time.monotonic() - wall_start < 55, 'cooperative wall budget')

    try:
        audit(report, deadline)
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - preserve completed readback prefix.
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        try:
            for path, digest in report['inputs_sha256'].items():
                require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest,
                        'input changed after audit: ' + path)
            report['verification_after'] = 'COMPLETE'
            report['source_and_artifact_hashes_reverified'] = len(report['inputs_sha256'])
        except BaseException as exc:  # noqa: BLE001 - late failure revokes audit completion.
            report['status'] = 'INCOMPLETE'
            report['verification_after'] = 'FAILED'
            report['verification_failure'] = type(exc).__name__ + ': ' + str(exc)
        report['cpu_seconds'] = time.process_time() - cpu_start
        report['wall_seconds'] = time.monotonic() - wall_start
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        require(len(text.encode()) + initial.stat().st_size <= CAP, '2MiB combined audit artifact cap')
        with args.output.open('x') as stream:
            stream.write(text)
        signal.alarm(0)
    print(json.dumps({key: report[key] for key in ('status', 'verification_after', 'rows_checked', 'cpu_seconds')}))
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
