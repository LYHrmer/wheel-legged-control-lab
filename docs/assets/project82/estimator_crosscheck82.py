"""Independent stdlib readback of all49/50 closed-loop preflight CSVs.

No experiment module is imported. Failed terminal episodes remain in the full
ledger; paired RMSE uses each original comparison's own common-full subset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import resource
import signal
import time
from collections import Counter
from pathlib import Path

CV = 'constant_velocity'
CANDIDATE = 'constant_velocity_measured_leg_q'
SEEDS = (17, 29, 43)
NOISES = ('noise', 'bias')
CAP = 2 << 20


def mean(values):
    return math.fsum(values) / len(values)


def rmse(values):
    return math.sqrt(mean([x*x for x in values]))


def identity(case):
    return case['noise'], case['seed'], case['arm']


def name(case):
    return f"flat_{case['noise']}_delay2_seed{case['seed']}_{case['arm']}"


class Audit:
    def __init__(self, root):
        self.root = root.resolve()
        self.started = time.process_time()
        self.inputs, self.gaps, self.csv_cache = {}, {}, {}
        self.cases, self.phases, self.sources = [], {}, []
        self.checks, self.rows_checked = 0, 0

    def limit(self):
        if time.process_time() - self.started >= 28:
            raise TimeoutError('cooperative28s CPU limit')

    def check(self, value, message):
        self.checks += 1
        if not value:
            raise ValueError(message)

    def close(self, actual, expected, label):
        self.check(math.isfinite(actual) and math.isfinite(expected), label+' nonfinite')
        gap = abs(actual-expected)
        self.gaps[label] = max(self.gaps.get(label, 0.), gap)
        self.check(gap <= 1e-11, label+' reconstruction differs')

    def bind(self, path, expected=None):
        self.limit()
        path = path.resolve()
        self.check(path.is_relative_to(self.root), 'input outside repository')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        old = self.inputs.setdefault(str(path), digest)
        self.check(old == digest, 'input changed while reading')
        if expected is not None:
            self.check(digest == expected, 'bound SHA differs: '+str(path))
        return data

    def js(self, path, expected=None):
        return json.loads(self.bind(path, expected))

    def compare(self, cases, baseline, candidate):
        by_id = {identity(c): c for c in cases}
        pairs = []
        for noise in NOISES:
            for seed in SEEDS:
                left, right = by_id[noise, seed, baseline], by_id[noise, seed, candidate]
                pairs.append({'noise': noise, 'seed': seed, 'baseline_arm': baseline,
                    'candidate_arm': candidate, 'baseline_full400': left['full400'],
                    'candidate_full400': right['full400'],
                    'baseline_termination': left['termination_reason'],
                    'candidate_termination': right['termination_reason'],
                    'baseline_steps': left['steps'], 'candidate_steps': right['steps'],
                    'baseline_rmse_mps': left['tracking_rmse_mps'],
                    'candidate_rmse_mps': right['tracking_rmse_mps'],
                    'common_full': left['full400'] and right['full400']})
        common = [p for p in pairs if p['common_full']]
        self.check(bool(common), 'original comparison has no common complete pair')
        left = mean([p['baseline_rmse_mps'] for p in common])
        right = mean([p['candidate_rmse_mps'] for p in common])
        self.check(left > 0, 'zero baseline mean RMSE')
        return {'baseline_arm': baseline, 'candidate_arm': candidate, 'all_six_pairs': pairs,
            'common_complete_pairs': len(common),
            'common_identities': [{'noise': p['noise'], 'seed': p['seed']} for p in common],
            'baseline_common_mean_rmse_mps': left, 'candidate_common_mean_rmse_mps': right,
            'candidate_over_baseline_ratio': right/left, 'change_percent': (right/left-1)*100,
            'aggregation': 'equal-weight mean of per-episode RMSE, original common-full pairs only'}

    def read_phase(self, number, date):
        folder = self.root/f'results/estimator{number}_preflight_{date}_01'
        publication = self.js(self.root/f'reports/estimator{number}_results_{date}.json')
        manifest = publication['artifact_sha256' if number == 49 else 'read_input_sha256']
        summary_path = folder/'summary.json'
        summary = self.js(summary_path, manifest[str(summary_path.relative_to(self.root))])
        prereg_path = self.root/f'docs/experiments{number}/estimator_latency{number}_preregistered.json'
        prereg = self.js(prereg_path, summary['preregistration_sha256'])
        self.check(summary['preregistration'] == prereg and summary['status'] == 'complete'
                   and summary['phase'] == 'preflight' and summary['source_closure_reverified_at_end'] is True,
                   'incomplete/unbound original phase')
        self.bind(self.root/f'scripts/estimator_candidate{number}.py', summary['script_sha256'])
        # Python sources only. Do not claim revalidation of all old asset/raw closures.
        for relative, expected in prereg['source_sha256'].items():
            if relative.endswith('.py') and relative.startswith(('src/', 'scripts/', 'tests/')):
                self.bind(self.root/relative, expected)
                self.sources.append({'phase': number, 'path': relative, 'sha256': expected,
                                     'current_equals_executed_binding': True})
        arms = ('none', CV) if number == 49 else ('none', CV, CANDIDATE)
        expected_ids = [(noise, seed, arm) for noise in NOISES for seed in SEEDS for arm in arms]
        cases = summary['cases']
        self.check([identity(c) for c in cases] == expected_ids
                   and [identity(c) for c in summary['planned_cases']] == expected_ids,
                   'registered full ordered case matrix differs')
        self.check({p.name for p in folder.glob('*.csv')} == {name(c)+'.csv' for c in cases},
                   'missing/extra preflight CSV')
        phase_rows = []
        self.phases[str(number)] = {'status': 'INCOMPLETE', 'cases_completed': 0}
        for case in cases:
            self.limit()
            self.check(case['terrain'] == 'flat' and case['delay_steps'] == 2
                       and case['termination_reason'] in ('time_limit', 'fallen')
                       and type(case['steps']) is int and 0 < case['steps'] <= 400,
                       'unsupported case identity/terminal status')
            stem = name(case)
            artifact = summary.get('output_artifact_sha256', {})
            json_rel = str((folder/(stem+'.json')).relative_to(self.root))
            case_expected = manifest[json_rel] if number == 49 else artifact[stem+'.json']
            saved_case = self.js(folder/(stem+'.json'), case_expected)
            for key in ('terrain', 'noise', 'seed', 'delay_steps', 'arm', 'steps',
                        'termination_reason', 'tracking_rmse_mps', 'velocity_estimation_rmse_mps',
                        'compensation_status_counts', 'replay_bitwise_equal', 'provenance_invariants_pass'):
                self.check(saved_case[key] == case[key], 'per-case terminal JSON/summary differs: '+key)
            if number == 50:
                self.check(saved_case['single_field_ablation_bitwise_pass'] is True
                           and case['single_field_ablation_bitwise_pass'] is True,
                           'phase50 single-field ablation integrity failed or differs')
            csv_path = folder/(stem+'.csv')
            csv_rel = str(csv_path.relative_to(self.root))
            data = self.bind(csv_path, manifest[csv_rel])
            if number == 50:
                self.check(hashlib.sha256(data).hexdigest() == artifact[stem+'.csv'], 'phase50 CSV seal differs')
            reader = csv.DictReader(io.StringIO(data.decode()))
            self.check(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                       'CSV missing/duplicate fields')
            rows = list(reader)
            self.check(len(rows) == case['steps'], 'CSV terminal length differs')
            numeric_names = [k for k in reader.fieldnames if k != 'applied_compensation_status']
            numeric = []
            statuses = Counter()
            for index, row in enumerate(rows):
                self.check(set(row) == set(reader.fieldnames) and None not in row.values(), 'malformed CSV row')
                values = {k: float(row[k]) for k in numeric_names}
                self.check(all(math.isfinite(v) for v in values.values()), 'nonfinite CSV')
                self.check(values['step'] == index+1, 'control index discontinuity')
                self.close(values['time_s'], (index+1)*.01, 'control_clock')
                age = values['time_s']-values['measurement_time_s']
                self.close(age, values['measurement_age_s'], 'post_state_measurement_age')
                self.close(age, min((index+1)*.01, .02), 'post_state_delay2_age')
                self.close(values['applied_measurement_age_s'], min(index*.01, .02), 'pre_state_delay2_age')
                fraction = max(0., min(1., (index*.01-.3)/.3))
                self.close(values['command_velocity_mps'], .35*fraction**2*(3-2*fraction), 'frozen_command')
                status = row['applied_compensation_status']
                statuses[status] += 1
                self.check(status in ('disabled', 'bypassed', 'applied', 'horizon_exceeded',
                                     'kinematic_horizon_exceeded'), 'unknown compensation status')
                self.close(values['applied_compensation_horizon_s'],
                           values['applied_measurement_age_s'] if status == 'applied' else 0., 'applied_horizon')
                numeric.append(values)
                self.rows_checked += 1
            self.check(dict(statuses) == case['compensation_status_counts'], 'compensation counts differ')
            tracking = rmse([r['truth_velocity_body_x_mps']-r['command_velocity_mps'] for r in numeric])
            velocity = rmse([r['estimate_velocity_body_x_mps']-r['truth_velocity_body_x_mps'] for r in numeric])
            self.close(tracking, case['tracking_rmse_mps'], 'tracking_rmse')
            self.close(velocity, case['velocity_estimation_rmse_mps'], 'velocity_estimation_rmse')
            self.close(case['duration_s'], len(rows)*.01, 'duration')
            self.check(case['native_steps'] == 5*len(rows), 'explicit substep count differs')
            self.check(case['replay_bitwise_equal'] is True and case['provenance_invariants_pass'] is True,
                       'original replay/provenance flags failed')
            result = {k: case[k] for k in ('terrain', 'noise', 'seed', 'delay_steps', 'arm',
                                          'steps', 'termination_reason')}
            result.update(phase=number, full400=case['steps'] == 400 and case['termination_reason'] == 'time_limit',
                tracking_rmse_mps=tracking, velocity_estimation_rmse_mps=velocity,
                compensation_status_counts=dict(statuses), csv_sha256=hashlib.sha256(data).hexdigest(),
                termination_source='complete per-case terminal JSON agrees with phase summary; CSV lacks roll/full xy')
            if number == 50:
                pd = [r['leg_pd_delta_from_measured_q_rms_nm'] for r in numeric]
                result['extra_unclipped_leg_pd_per_row_rms_nm'] = {'max': max(pd), 'mean': mean(pd),
                    'all_zero': all(v == 0 for v in pd)}
                if case['arm'] in ('none', CANDIDATE):
                    self.check(all(v == 0 for v in pd), 'measured leg q has nonzero added PD RMS')
            self.csv_cache[number, identity(case)] = (reader.fieldnames, rows)
            self.cases.append(result)
            phase_rows.append(result)
            self.phases[str(number)]['cases_completed'] += 1
        counts = {arm: {'cases': sum(r['arm'] == arm for r in phase_rows),
            'full400': sum(r['arm'] == arm and r['full400'] for r in phase_rows),
            'fallen': sum(r['arm'] == arm and r['termination_reason'] == 'fallen' for r in phase_rows),
            'control_rows': sum(r['steps'] for r in phase_rows if r['arm'] == arm)} for arm in arms}
        candidate = CV if number == 49 else CANDIDATE
        comparison = self.compare(phase_rows, 'none', candidate)
        gate = summary['gate']
        for key, computed in (('common_raw_tracking_rmse_mps', comparison['baseline_common_mean_rmse_mps']),
                              ('common_candidate_tracking_rmse_mps', comparison['candidate_common_mean_rmse_mps']),
                              ('common_tracking_rmse_ratio', comparison['candidate_over_baseline_ratio'])):
            self.close(computed, gate[key], 'original_gate_'+key)
        self.check(comparison['common_complete_pairs'] == gate['common_complete_pairs'], 'original common set count')
        self.check({a: c['full400'] for a, c in counts.items()} == gate['completions']['2'], 'completion count')
        for arm, c in counts.items():
            self.check({'time_limit': c['full400'], 'fallen': c['fallen']} == gate['termination_counts_by_arm'][arm],
                       'termination counts')
        gain = counts[candidate]['full400']-counts['none']['full400']
        no_extra_falls = counts[candidate]['fallen'] <= counts['none']['fallen']
        ratio = comparison['candidate_over_baseline_ratio']
        integrity = all(c['replay_bitwise_equal'] is True and c['provenance_invariants_pass'] is True
                        and (number == 49 or c['single_field_ablation_bitwise_pass'] is True) for c in cases)
        self.check(integrity == gate['replay_and_provenance_pass'], 'original aggregate integrity differs')
        computed_pass = integrity and no_extra_falls and gain >= 0 and ratio <= 1.05 and (gain > 0 or ratio <= .98)
        self.check(computed_pass == gate['passed'] and gain == gate['delay20_completion_gain']
                   and no_extra_falls == gate['no_extra_falls'], 'original preflight decision differs')
        self.check(sorted(name(c) for c in phase_rows if not c['full400']) == sorted(gate['failed_cases']),
                   'failed cases omitted')
        controls = sum(r['steps'] for r in phase_rows)
        self.check(controls == summary['executed_control_steps'] and 5*controls == summary['executed_native_steps'],
                   'phase executed counts differ')
        self.check(publication['full']['status'].lower() == 'not_run', 'full matrix unexpectedly ran')
        phase = {'status': 'COMPLETE', 'cases_completed': len(phase_rows), 'partial_cases': 0,
            'counts_by_arm': counts, 'control_rows': controls, 'explicit_control_substeps': controls*5,
            'performance_comparison': comparison, 'original_gate_recomputed': bool(computed_pass),
            'replay_and_provenance_integrity_recomputed': integrity,
            'original_gate': gate, 'full_matrix_original_status': publication['full']['status'],
            'current_observer_binding': prereg['source_sha256']['src/wheel_legged_control/d1/sensor_estimation.py'],
            'old_archive_observer_sha256': prereg['archive_observer_source_sha256'],
            'declared_archive_source_difference': prereg['archive_source_difference']}
        if number == 50:
            mechanism = self.compare(phase_rows, CV, CANDIDATE)
            reference = gate['mechanism_reference']
            for key, computed in (('cv_mean_tracking_rmse_mps', mechanism['baseline_common_mean_rmse_mps']),
                                  ('candidate_mean_tracking_rmse_mps', mechanism['candidate_common_mean_rmse_mps']),
                                  ('candidate_over_cv_tracking_ratio', mechanism['candidate_over_baseline_ratio'])):
                self.close(computed, reference[key], 'mechanism_'+key)
            self.check(mechanism['common_complete_pairs'] == reference['common_complete_pairs'], 'mechanism common count')
            mechanism['same_subset_as_performance'] = mechanism['common_identities'] == comparison['common_identities']
            mechanism['used_for_performance_gate'] = False
            phase['mechanism_comparison'] = mechanism
            phase['none_vs_cv_old_comparison_repeated_on50'] = self.compare(phase_rows, 'none', CV)
        self.phases[str(number)] = phase

    def run(self):
        self.read_phase(49, '20261022')
        self.read_phase(50, '20261024')
        reproduction = []
        for noise in NOISES:
            for seed in SEEDS:
                for arm in ('none', CV):
                    key = noise, seed, arm
                    names49, rows49 = self.csv_cache[49, key]
                    names50, rows50 = self.csv_cache[50, key]
                    self.check(set(names49) <= set(names50) and len(rows49) == len(rows50), 'cross-phase CSV scope')
                    for left, right in zip(rows49, rows50, strict=True):
                        for field in names49:
                            if field == 'applied_compensation_status':
                                self.check(left[field] == right[field], 'cross-phase status differs')
                            else:
                                self.check(float(left[field]) == float(right[field]), 'cross-phase numeric value differs')
                    reproduction.append({'noise': noise, 'seed': seed, 'arm': arm, 'rows': len(rows49),
                        'common_fields': names49, 'all_common_values_exact': True})
        old = self.phases['49']['performance_comparison']
        repeated = self.phases['50']['none_vs_cv_old_comparison_repeated_on50']
        self.check(old == repeated, 'same original none/CV comparison changed across phases')
        for path, digest in self.inputs.items():
            self.limit()
            self.check(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'input changed after readback')
        return {**self.partial(), 'status': 'COMPLETE', 'verification_after': 'COMPLETE',
            'cross_phase_reproduction_12_cases': reproduction,
            'source_binding_scope': 'all registered Python source files plus both entry scripts; not a repeat hash of all old assets/NPZ closures',
            'source_notes': {
                'observer': 'd1/sensor_estimation.py:195,294,412: default wheel_center_jacobians=False; synchronized opts into mj_jacBody, legacy uses mj_jacBodyCom.49/50 fixed legacy_mixed.',
                'guard': 'd1/state_estimation.py:437-492: zero-age bypass; age>0.05+1e-12 rejects; leg delta>0.35 or predicted leg-limit violation rejects before any candidate50 leg-q restoration.',
                'cv_changed': 'state_estimation.py:495-529: base position/rotation, world linear/angular velocity, joint position, foot position/Jacobian; body-frame velocities, joint velocity and measured contact/time provenance remain.',
                'candidate50': 'scripts/estimator_candidate50.py:113-166: full CV first; only applied 12 leg joint_position entries restored to raw; wheel q and every other CV field retained; no whole-state current-time claim.'},
            'interpretation_limits': [
                'Old144 source SHA differs from current observer. Its17/36 falls remain historical evidence, not a newly reproduced current full matrix.',
                '49 and50 use current-bound source with legacy_mixed sampling. This does not validate the opt-in synchronized path.',
                'All30 registered closed-loop terminal episodes are retained, including10 falls; terminal failures are not interrupted partial runs.',
                'Each comparison retains its original common-full subset. Mechanism CV comparison cannot replace candidate versus none.',
                'CSV terminal length is independently checked; fall reasons come from matching terminal JSON/summary because CSV lacks roll and full world xy.',
                'CSV added-PD RMS is a pre-limit component; it is not final actuator torque or the maximum single-axis value.',
                'No per-wheel residual, new model, controller rollout or causal diagnosis is produced.',
                'Control rows times5 count explicit control integration only, not every compiler/reset/kinematic native call.']}

    def partial(self):
        return {'schema': 'd1-estimator49-50-crosscheck82-v1', 'status': 'INCOMPLETE',
            'verification_after': 'NOT_COMPLETE', 'checks': self.checks, 'csv_rows_checked': self.rows_checked,
            'cases_completed': len(self.cases), 'cases': self.cases, 'phases': self.phases,
            'inputs_sha256': self.inputs, 'source_bindings_checked': self.sources,
            'maximum_reconstruction_gaps': self.gaps,
            'new_model_calls': 0, 'new_physics_steps': 0, 'new_parameter_updates': 0,
            'original_gates_changed': False, 'partial_cases_cannot_qualify': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/lyh/wheel-legged-control-lab'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    resource.setrlimit(resource.RLIMIT_CPU, (29, 30))

    def stop(signum, _frame):
        raise TimeoutError('readback supervisor signal '+str(signum))

    for sig in (signal.SIGXCPU, signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    signal.alarm(60)
    start = time.monotonic()
    audit = Audit(args.root)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        result = audit.run()
        audit.limit()
        audit.check(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == source_sha, 'audit script changed')
    except BaseException as exc:  # noqa: BLE001 - retain partial readback, never promote it.
        result = {**audit.partial(), 'error': f'{type(exc).__name__}: {exc}'}
    result.update(cpu_seconds=time.process_time()-audit.started, wall_seconds=time.monotonic()-start,
        script_sha256=source_sha, limits={'cpu_soft_s': 29, 'cpu_hard_s': 30,
                                        'wall_alarm_s': 60, 'output_cap_bytes': CAP})
    payload = json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n'
    if len(payload.encode()) > CAP:
        result = {'schema': 'd1-estimator49-50-crosscheck82-v1', 'status': 'INCOMPLETE',
            'error': 'output exceeds2MiB', 'cases_completed': len(audit.cases),
            'csv_rows_checked': audit.rows_checked, 'new_model_calls': 0,
            'new_physics_steps': 0, 'new_parameter_updates': 0, 'original_gates_changed': False}
        payload = json.dumps(result, allow_nan=False)+'\n'
    with args.output.open('x') as stream:
        stream.write(payload)
    signal.alarm(0)
    print(json.dumps({k: result.get(k) for k in ('status', 'cases_completed', 'csv_rows_checked',
                                               'cpu_seconds', 'error')}, allow_nan=False))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
