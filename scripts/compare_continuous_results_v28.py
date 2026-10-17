"""Compare audited continuous-policy metrics on complete, equal task horizons.

Reads only small protocol/manifest/audit/CSV files with stdlib. It does not
recompute raw trajectories, import project models, or grant qualification.
An incomplete or missing pair has null deltas; no survivor-only group mean
is substituted for a predeclared holdout matrix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

FIELDS = ('velocity_error_mps', 'lateral_velocity_mps', 'yaw_rate_error_rps',
          'clearance_error_m', 'lateral_path_error_m', 'longitudinal_path_error_m')
PROTOCOL_SCHEMA = 'd1-continuous-sensor-ppo-experiment-v1'
AUDIT_SCHEMA = 'd1-continuous-policy-independent-audit-v1'
SCHEMA = 'd1-continuous-comparison-v28'
TIME_TOLERANCE_S = 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def number(value, label):
    require(not isinstance(value, bool), f'{label}: boolean number')
    result = float(value)
    require(math.isfinite(result) and result >= 0, f'{label}: expected finite nonnegative number')
    return result


def integer(value, label):
    require(type(value) is int or (type(value) is str and value.isascii() and value.isdigit()),
            f'{label}: expected nonnegative integer')
    result = int(value)
    require(result >= 0, f'{label}: expected nonnegative integer')
    return result


def boolean(value, label):
    if type(value) is bool:
        return value
    require(value in ('True', 'False'), f'{label}: expected True or False')
    return value == 'True'


def expected_matrix(protocol):
    require(type(protocol) is dict, 'protocol must be an object')
    require(protocol.get('schema') == PROTOCOL_SCHEMA and protocol.get('smoke_only') is False,
            'unsupported or smoke-only continuous protocol')
    require(type(protocol.get('evaluation_cases')) is dict
            and set(protocol['evaluation_cases']) == {'development', 'holdout'},
            'unsupported evaluation split matrix')
    seeds, budgets = protocol['training_seeds'], protocol['checkpoint_budgets']
    require(type(seeds) is list and seeds and all(type(seed) is int and seed >= 0 for seed in seeds)
            and len(set(seeds)) == len(seeds), 'invalid or duplicate training seeds')
    require(type(budgets) is list and len(budgets) >= 2
            and all(type(budget) is int and budget > 0 for budget in budgets)
            and budgets == sorted(set(budgets)), 'invalid checkpoint budgets')
    cases, all_cases = {}, {}
    for split in ('development', 'holdout'):
        rows = protocol['evaluation_cases'][split]
        require(type(rows) is list and rows, f'empty evaluation cases: {split}')
        cases[split] = []
        for case in rows:
            seed = integer(case['seed'], 'case seed')
            duration = number(case['duration_s'], 'case duration')
            require(duration > 0, 'nonpositive case duration')
            name = f'seed{seed}_{duration:g}s'
            require(name not in all_cases, 'duplicate/overlapping development or holdout case')
            all_cases[name] = duration
            cases[split].append(name)
    require(len(cases['development']) >= 2, 'development checkpoint rule needs two cases')
    expected = {}
    for case, duration in all_cases.items():
        name = 'zero_'+case
        expected[name] = {'rollout': name, 'mode': 'zero', 'case': case, 'training_seed': 0,
                          'budget': 0, 'split': 'baseline', 'expected_duration_s': duration}
    for seed in seeds:
        for budget in budgets:
            selected = (cases['development'] if budget == budgets[-1] else
                        [cases['development'][0 if budget == budgets[0] else 1]])
            for case in selected:
                name = f'dev_train{seed}_step{budget}_{case}'
                expected[name] = {'rollout': name, 'mode': 'policy', 'case': case,
                                  'training_seed': seed, 'budget': budget, 'split': 'development',
                                  'expected_duration_s': all_cases[case]}
        for case in cases['holdout']:
            name = f'holdout_train{seed}_{case}'
            expected[name] = {'rollout': name, 'mode': 'policy', 'case': case,
                              'training_seed': seed, 'budget': budgets[-1], 'split': 'holdout',
                              'expected_duration_s': all_cases[case]}
    return expected


def read_metrics(data):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
    required = {'rollout', 'mode', 'case', 'training_seed', 'budget', 'split', 'completed',
                'quality_pass', 'duration_s', *FIELDS}
    require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames)
            and required <= set(reader.fieldnames), 'missing or duplicate CSV columns')
    rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows),
            'empty or malformed metrics CSV')
    return rows


def compare_rows(protocol, raw_rows):
    expected = expected_matrix(protocol)
    indexed = {}
    for raw in raw_rows:
        name = raw['rollout']
        require(name in expected and name not in indexed, f'unexpected or duplicate rollout: {name}')
        row = {key: raw[key] for key in ('rollout', 'mode', 'case', 'split')}
        row.update({key: integer(raw[key], key) for key in ('training_seed', 'budget')})
        row.update({key: boolean(raw[key], key) for key in ('completed', 'quality_pass')})
        row.update({key: number(raw[key], key) for key in ('duration_s', *FIELDS)})
        for key in ('mode', 'case', 'split', 'training_seed', 'budget'):
            require(row[key] == expected[name][key], f'rollout identity mismatch: {name}: {key}')
        require(not row['quality_pass'] or row['completed'], 'quality pass on incomplete episode')
        indexed[name] = row
    # Keep existing CSV order for exact agreement with the original six-case means.
    policy_names = [name for name, row in indexed.items() if row['mode'] == 'policy']
    policy_names += [name for name, row in expected.items() if row['mode'] == 'policy' and name not in indexed]
    pairs = []
    for name in policy_names:
        plan, policy = expected[name], indexed.get(name)
        baseline = indexed.get('zero_'+plan['case'])
        reasons = []
        for label, row in (('policy', policy), ('baseline', baseline)):
            if row is None:
                reasons.append('missing_'+label)
                continue
            if not row['completed']:
                reasons.append(label+'_incomplete')
            if abs(row['duration_s']-plan['expected_duration_s']) > TIME_TOLERANCE_S:
                reasons.append(label+'_duration_mismatch')
        pair = {key: plan[key] for key in ('rollout', 'training_seed', 'budget', 'split', 'case', 'expected_duration_s')}
        pair.update(status='INCOMPARABLE' if reasons else 'COMPARABLE', reasons=reasons)
        for label, row in (('policy', policy), ('baseline', baseline)):
            for key in ('completed', 'quality_pass', 'duration_s'):
                pair[label+'_'+key] = None if row is None else row[key]
        pair['deltas'] = None if reasons else {field: policy[field]-baseline[field] for field in FIELDS}
        pairs.append(pair)
    groups = []
    for seed in protocol['training_seeds']:
        selected = [pair for pair in pairs if pair['training_seed'] == seed and pair['split'] == 'holdout']
        comparable = [pair for pair in selected if pair['status'] == 'COMPARABLE']
        groups.append({'training_seed': seed, 'expected_cases': len(selected),
                       'present_policy_cases': sum(pair['policy_completed'] is not None for pair in selected),
                       'comparable_cases': len(comparable),
                       'policy_complete': sum(pair['policy_completed'] is True for pair in selected),
                       'baseline_complete': sum(pair['baseline_completed'] is True for pair in selected),
                       'policy_quality': sum(pair['policy_quality_pass'] is True for pair in selected),
                       'baseline_quality': sum(pair['baseline_quality_pass'] is True for pair in selected),
                       'missing_cases': [pair['case'] for pair in selected if any(
                           reason.startswith('missing_') for reason in pair['reasons'])],
                       'invalid_cases': [{'case': pair['case'], 'reasons': pair['reasons']}
                                         for pair in selected if pair['reasons']],
                       'status': 'COMPARABLE' if len(comparable) == len(selected) else 'INCOMPARABLE',
                       'full_horizon_mean_deltas': {field: sum(pair['deltas'][field] for pair in selected)/len(selected)
                                                    for field in FIELDS}
                       if len(comparable) == len(selected) else None})
    invalid = sum(pair['status'] != 'COMPARABLE' for pair in pairs)
    return {'schema': SCHEMA, 'status': 'INCOMPARABLE' if invalid else 'COMPARABLE',
            'expected_rollouts': len(expected), 'observed_rollouts': len(indexed),
            'missing_rollouts': sorted(set(expected)-set(indexed)), 'pairs': pairs,
            'pair_count': len(pairs), 'holdout_pair_count': sum(pair['split'] == 'holdout' for pair in pairs),
            'invalid_pair_count': invalid, 'holdout_by_training_seed': groups,
            'scope': 'comparison of previously audited metrics; raw trajectories are not re-audited',
            'replication_unit': 'training seed; no pooled-step inference or confidence interval',
            'models_loaded': 0, 'gui_starts': 0, 'physics_steps': 0, 'qualification_granted': False}


def load_inputs(run, audit):
    paths = {'run_manifest': run/'manifest.json', 'protocol': run/'protocol.json',
             'audit_manifest': audit/'manifest.json', 'audit_report': audit/'independent_audit.json',
             'metrics': audit/'metrics_recomputed.csv'}
    payloads = {key: path.read_bytes() for key, path in paths.items()}
    run_manifest, audit_manifest, witness, protocol = (
        strict_json(payloads[key]) for key in ('run_manifest', 'audit_manifest', 'audit_report', 'protocol'))
    require(run_manifest['sha256']['protocol.json'] == sha(payloads['protocol']), 'protocol SHA mismatch')
    require(audit_manifest['sha256']['independent_audit.json'] == sha(payloads['audit_report']),
            'audit report SHA mismatch')
    require(audit_manifest['sha256']['metrics_recomputed.csv'] == sha(payloads['metrics']), 'metrics SHA mismatch')
    require(witness['schema'] == AUDIT_SCHEMA
            and witness['input_manifest_sha256'] == sha(payloads['run_manifest'])
            and witness.get('complete_case_inventory_matches_protocol') is True,
            'audit report is not bound to this experiment manifest and case inventory')
    bindings = {key: {'path': str(paths[key]), 'bytes': len(data), 'sha256': sha(data)}
                for key, data in payloads.items()}
    return protocol, read_metrics(payloads['metrics']), bindings, paths


def run(run_dir, audit_dir, output):
    run_dir, audit_dir, output = (Path(path).resolve() for path in (run_dir, audit_dir, output))
    require(not output.exists(), 'output must be a new directory')
    require(not output.is_relative_to(run_dir) and not output.is_relative_to(audit_dir),
            'output must be outside immutable run/audit inputs')
    protocol, metrics, bindings, paths = load_inputs(run_dir, audit_dir)
    report = compare_rows(protocol, metrics)
    require(all(sha(path.read_bytes()) == bindings[key]['sha256'] for key, path in paths.items()),
            'input changed during comparison')
    report.update(input_identities=bindings, entry_sha256=sha(Path(__file__).read_bytes()))
    output.mkdir(parents=True, exist_ok=False)
    pairs = []
    for pair in report['pairs']:
        row = {key: value for key, value in pair.items() if key not in ('deltas', 'reasons')}
        row['reasons'] = ';'.join(pair['reasons'])
        row.update({'delta_'+field: None if pair['deltas'] is None else pair['deltas'][field] for field in FIELDS})
        pairs.append(row)
    with (output/'paired_cases.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    # Completion marker follows the exclusive CSV write; partial outputs are retained.
    with (output/'comparison.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.run, args.audit, args.output)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, csv.Error) as exc:
        print(json.dumps({'status': 'ERROR', 'error': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in ('status', 'observed_rollouts', 'pair_count',
          'holdout_pair_count', 'invalid_pair_count', 'missing_rollouts', 'models_loaded', 'gui_starts', 'physics_steps')}))
    return 0 if report['status'] == 'COMPARABLE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
