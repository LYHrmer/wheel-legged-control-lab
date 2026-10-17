"""Complete-horizon pairing and small-file identity chains; no model or physics."""
import copy
import csv
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/compare_continuous_results_v28.py'
_spec = importlib.util.spec_from_file_location('continuous_compare_v28', SCRIPT)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def protocol():
    return {'schema': tool.PROTOCOL_SCHEMA, 'smoke_only': False,
            'training_seeds': [19000], 'checkpoint_budgets': [32768, 65536, 131072],
            'evaluation_cases': {
                'development': [{'seed': seed, 'duration_s': 45.} for seed in (17, 29, 43)],
                'holdout': [{'seed': 617, 'duration_s': duration} for duration in (45., 50.)]}}


def metrics(config):
    return [{**{key: value for key, value in plan.items() if key != 'expected_duration_s'},
             'completed': True, 'quality_pass': False, 'duration_s': plan['expected_duration_s'],
             **dict.fromkeys(tool.FIELDS, .1 if plan['mode'] == 'zero' else .08)}
            for plan in tool.expected_matrix(config).values()]


def holdout(rows):
    return next(row for row in rows if row['split'] == 'holdout')


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def put_metrics(path, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue())


def refresh_audit_manifest(audit):
    put_json(audit/'manifest.json', {'sha256': {name: tool.sha((audit/name).read_bytes())
             for name in ('metrics_recomputed.csv', 'independent_audit.json')}})


@pytest.fixture
def inputs(tmp_path):
    run, audit = tmp_path/'experiment', tmp_path/'audit'
    run.mkdir()
    audit.mkdir()
    config = protocol()
    put_json(run/'protocol.json', config)
    put_json(run/'manifest.json', {'sha256': {'protocol.json': tool.sha((run/'protocol.json').read_bytes())}})
    put_metrics(audit/'metrics_recomputed.csv', metrics(config))
    put_json(audit/'independent_audit.json', {'schema': tool.AUDIT_SCHEMA,
             'input_manifest_sha256': tool.sha((run/'manifest.json').read_bytes()),
             'complete_case_inventory_matches_protocol': True})
    refresh_audit_manifest(audit)
    return run, audit


def test_old_failed_short_episode_produces_a_delta_but_new_pair_is_incomparable():
    legacy_spec = importlib.util.spec_from_file_location('legacy_continuous_pair_test',
                                                       ROOT/'scripts/analyze_d1_continuous_policy.py')
    legacy = importlib.util.module_from_spec(legacy_spec)
    legacy_spec.loader.exec_module(legacy)
    config = protocol()
    rows = metrics(config)
    failed = holdout(rows)
    zero = next(row for row in rows if row['mode'] == 'zero' and row['case'] == failed['case'])
    failed.update(completed=False, duration_s=1.)
    failed.update(dict.fromkeys(tool.FIELDS, 0.))
    old_pair = legacy.paired_rows([zero, failed])[0]
    assert old_pair['delta_velocity_error_mps'] == -.1  # Reproduce before checking v28.
    report = tool.compare_rows(config, rows)
    pair = next(pair for pair in report['pairs'] if pair['rollout'] == failed['rollout'])
    assert pair['status'] == 'INCOMPARABLE' and pair['deltas'] is None
    assert pair['policy_duration_s'] == 1. and pair['baseline_duration_s'] == 45.
    assert report['holdout_by_training_seed'][0]['full_horizon_mean_deltas'] is None


@pytest.mark.parametrize('change', ('policy_incomplete', 'baseline_incomplete', 'policy_short',
                                   'baseline_short', 'policy_long'))
def test_completion_flags_and_equal_expected_horizons_are_both_required(change):
    config = protocol()
    rows = metrics(config)
    policy = holdout(rows)
    baseline = next(row for row in rows if row['mode'] == 'zero' and row['case'] == policy['case'])
    target = policy if change.startswith('policy') else baseline
    if change.endswith('incomplete'):
        target['completed'] = False
    else:
        target['duration_s'] += 1. if change.endswith('long') else -1.
    report = tool.compare_rows(config, rows)
    pair = next(pair for pair in report['pairs'] if pair['rollout'] == policy['rollout'])
    assert pair['deltas'] is None and report['invalid_pair_count'] > 0
    assert report['holdout_by_training_seed'][0]['full_horizon_mean_deltas'] is None


@pytest.mark.parametrize('remove', ('policy', 'baseline'))
def test_missing_predeclared_case_is_visible_and_never_becomes_a_survivor_mean(remove):
    config = protocol()
    rows = metrics(config)
    policy = holdout(rows)
    name = policy['rollout'] if remove == 'policy' else 'zero_'+policy['case']
    report = tool.compare_rows(config, [row for row in rows if row['rollout'] != name])
    group = report['holdout_by_training_seed'][0]
    assert report['status'] == 'INCOMPARABLE' and name in report['missing_rollouts']
    assert policy['case'] in group['missing_cases']
    assert group['comparable_cases'] == 1 and group['expected_cases'] == 2
    assert group['full_horizon_mean_deltas'] is None


def test_real_zero_improvement_is_distinct_from_incomparable_null():
    config = protocol()
    rows = metrics(config)
    for row in rows:
        row.update(dict.fromkeys(tool.FIELDS, .1))
    report = tool.compare_rows(config, rows)
    assert report['status'] == 'COMPARABLE' and report['invalid_pair_count'] == 0
    assert all(pair['deltas'] == dict.fromkeys(tool.FIELDS, 0.) for pair in report['pairs'])
    assert report['holdout_by_training_seed'][0]['full_horizon_mean_deltas'] == dict.fromkeys(tool.FIELDS, 0.)


@pytest.mark.parametrize('fault', ('duplicate', 'identity', 'nan', 'boolean_count', 'unknown_rollout'))
def test_malformed_or_wrongly_labeled_metric_rows_are_rejected(fault):
    config = protocol()
    rows = metrics(config)
    if fault == 'duplicate':
        rows.append(copy.deepcopy(rows[-1]))
    elif fault == 'identity':
        rows[-1]['training_seed'] = 20000
    elif fault == 'nan':
        rows[-1][tool.FIELDS[0]] = float('nan')
    elif fault == 'boolean_count':
        rows[-1]['budget'] = True
    else:
        rows[-1]['rollout'] = 'unplanned_run'
    with pytest.raises(ValueError):
        tool.compare_rows(config, rows)


@pytest.mark.parametrize('change', ('protocol', 'metrics', 'audit_report', 'run_manifest', 'cross_experiment'))
def test_input_identity_chain_rejects_replaced_protocol_metrics_or_audit(inputs, tmp_path, change):
    run, audit = inputs
    if change == 'protocol':
        (run/'protocol.json').write_text('{}')
    elif change == 'metrics':
        with (audit/'metrics_recomputed.csv').open('a') as stream:
            stream.write('\n')
    elif change == 'audit_report':
        (audit/'independent_audit.json').write_text('{}')
    elif change == 'run_manifest':
        manifest = json.loads((run/'manifest.json').read_text())
        manifest['other_experiment'] = True
        put_json(run/'manifest.json', manifest)
    else:
        witness = json.loads((audit/'independent_audit.json').read_text())
        witness['input_manifest_sha256'] = '0'*64
        put_json(audit/'independent_audit.json', witness)
        refresh_audit_manifest(audit)
    output = tmp_path/'comparison'
    with pytest.raises(ValueError):
        tool.run(run, audit, output)
    assert not output.exists()


def test_cli_runs_without_site_packages_and_keeps_inputs_unchanged(inputs, tmp_path):
    run, audit = inputs
    paths = [*run.iterdir(), *audit.iterdir()]
    before = {str(path): path.read_bytes() for path in paths}
    output = tmp_path/'comparison'
    result = subprocess.run([sys.executable, '-S', '-B', str(SCRIPT), '--run', str(run),
                             '--audit', str(audit), '--output', str(output)],
                            check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output/'comparison.json').read_text())
    assert report['status'] == 'COMPARABLE' and report['physics_steps'] == 0
    assert report['models_loaded'] == report['gui_starts'] == 0
    assert len(report['input_identities']) == 5
    assert all(path.read_bytes() == before[str(path)] for path in paths)
    assert tool.main(['--run', str(run), '--audit', str(audit), '--output', str(output)]) == 2


def test_incomparable_cli_writes_null_deltas_and_returns_one(inputs, tmp_path):
    run, audit = inputs
    config = protocol()
    rows = metrics(config)
    holdout(rows).update(completed=False, duration_s=1.)
    put_metrics(audit/'metrics_recomputed.csv', rows)
    refresh_audit_manifest(audit)
    output = tmp_path/'comparison'
    assert tool.main(['--run', str(run), '--audit', str(audit), '--output', str(output)]) == 1
    report = json.loads((output/'comparison.json').read_text())
    assert report['invalid_pair_count'] == 1
    assert report['holdout_by_training_seed'][0]['full_horizon_mean_deltas'] is None
    parsed = list(csv.DictReader((output/'paired_cases.csv').open()))
    incomplete = next(row for row in parsed if row['status'] == 'INCOMPARABLE')
    assert incomplete['delta_velocity_error_mps'] == ''


def test_output_must_not_modify_run_or_audit_directory(inputs):
    run, audit = inputs
    for output in (run/'new_comparison', audit/'new_comparison'):
        with pytest.raises(ValueError, match='immutable'):
            tool.run(run, audit, output)
        assert not output.exists()


def test_archived_33_pairs_and_18_holdout_group_values_remain_exact():
    run = ROOT/'results/d1_continuous_policy/sensor45_v1'
    audit = ROOT/'results/d1_continuous_policy/analysis_sensor45_v1'
    if not all((audit/name).is_file() for name in ('metrics_recomputed.csv', 'independent_audit.json',
                                                'manifest.json', 'paired_cases.csv',
                                                'holdout_by_training_seed.csv')):
        pytest.skip('the historical audited-result bundle is not installed')
    config, rows, _, _ = tool.load_inputs(run, audit)
    report = tool.compare_rows(config, rows)
    assert report['pair_count'] == 33 and report['holdout_pair_count'] == 18
    assert report['invalid_pair_count'] == 0 and report['missing_rollouts'] == []
    inventory = json.loads((audit/'manifest.json').read_text())['sha256']
    for name in ('paired_cases.csv', 'holdout_by_training_seed.csv'):
        assert tool.sha((audit/name).read_bytes()) == inventory[name]
    key = lambda pair: (int(pair['training_seed']), int(pair['budget']), pair['split'], pair['case'])
    stored = {key(row): row for row in csv.DictReader((audit/'paired_cases.csv').open())}
    assert {key(pair) for pair in report['pairs']} == set(stored)
    for pair in report['pairs']:
        for field in tool.FIELDS:
            assert pair['deltas'][field] == float(stored[key(pair)]['delta_'+field])
    groups = {int(row['training_seed']): row for row in
              csv.DictReader((audit/'holdout_by_training_seed.csv').open())}
    for group in report['holdout_by_training_seed']:
        assert group['expected_cases'] == group['comparable_cases'] == 6
        for field in tool.FIELDS:
            assert group['full_horizon_mean_deltas'][field] == float(
                groups[group['training_seed']]['mean_paired_delta_'+field])
