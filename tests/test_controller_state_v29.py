"""Small synthetic telemetry proves diagnostic gaps without model or physics."""
import csv
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/diagnose_controller_state_v29.py'
_spec = importlib.util.spec_from_file_location('controller_diagnostic_v29', SCRIPT)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)
STAGES = [['left', 0., 22.5, 0., .08], ['right', 22.5, 45., 0., -.08]]


def telemetry(duration=.08):
    count = round(duration/tool.DT)
    rows = []
    for index in range(count):
        t = index*tool.DT*45./duration
        plan = next(stage for stage in STAGES if t < stage[2])
        rows.append({'step': index+1, 'time_s': (index+1)*tool.DT, 'stage': plan[0],
                     'stage_time_s': (t-plan[1])*duration/45.+tool.DT,
                     'terminated': 0, 'truncated': int(index == count-1),
                     'termination_reason': 'time_limit' if index == count-1 else 'ongoing',
                     'yaw_integral_nm': 1. if index % 4 < 2 else -1.,
                     'command_yaw_rate_rps': plan[4], 'yaw_differential_torque_nm': .2,
                     'yaw_rate_error_rps': .04, 'clearance_error_m': -.02,
                     'action_longitudinal': .1, 'action_vertical': 1.,
                     'residual_longitudinal_n': 1.125, 'residual_vertical_n': 20.,
                     'torque_saturation_fraction': .25 if index % 2 == 0 else 0.})
    return rows


def put_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def summarize(path):
    return tool.summarize_telemetry(path, .08, STAGES, [11.25, 20.])[0]


def test_zero_overall_integral_does_not_hide_opposed_integral_in_right_turn(tmp_path):
    rows = telemetry()
    # Whole-episode means alone cannot expose this recorded state carryover.
    assert sum(row['yaw_integral_nm'] for row in rows)/len(rows) == 0
    path = tmp_path/'telemetry.csv'
    put_csv(path, rows)
    result = summarize(path)
    right = next(row for row in result['stage_windows'] if row['stage'] == 'right'
                 and row['window'] == 'entry_first_1s')
    assert right['yaw_integral_nm']['mean'] == 0
    assert right['yaw_integral_nm']['first'] == 1
    assert right['integral_opposes_command'] == {
        'ticks': 2, 'denominator_commanded_yaw_ticks': 4, 'fraction': .5}
    assert right['ticks_with_any_recorded_torque_saturation'] == {
        'ticks': 2, 'denominator_ticks': 4, 'fraction': .5}
    assert right['vertical_action_at_clip']['fraction'] == 1


def test_missing_integral_is_unknown_and_route_reference_is_not_lqr_memory(tmp_path):
    rows = telemetry()
    for row in rows:
        del row['yaw_integral_nm']
        row.update(reference_x_m=0., reference_y_m=0., reference_yaw_rad=0.)
    path = tmp_path/'telemetry.csv'
    put_csv(path, rows)
    result = summarize(path)
    assert result['field_inventory']['yaw_integral_nm'] == 'NOT_RECORDED'
    assert result['field_inventory']['lqr_distance_reference_m'] == 'NOT_RECORDED'
    assert result['field_inventory']['route_reference_x_y_yaw'] == 'RECORDED_ROUTE_REFERENCE_NOT_LQR_MEMORY'
    assert all(row['yaw_integral_nm'] is None and row['integral_opposes_command'] is None
               for row in result['stage_windows'])


def test_zero_yaw_command_and_empty_late_window_have_no_fake_zero_ratio(tmp_path):
    rows = telemetry()
    for row in rows:
        row['command_yaw_rate_rps'] = 0.
    path = tmp_path/'telemetry.csv'
    put_csv(path, rows)
    windows = summarize(path)['stage_windows']
    assert all(row['integral_opposes_command']['fraction'] is None for row in windows)
    assert all(row['ticks'] == 0 and row['yaw_rate_error_rmse_rps'] is None
               for row in windows if row['window'] == 'after_1s')


def test_entry_window_uses_actual_stage_clock_and_every_tick_has_one_denominator(tmp_path):
    path = tmp_path/'telemetry.csv'
    put_csv(path, telemetry(2.4))
    result, _ = tool.summarize_telemetry(path, 2.4, STAGES, [11.25, 20.])
    assert [window['ticks'] for window in result['stage_windows']] == [100, 20, 100, 20]
    assert sum(window['ticks'] for window in result['stage_windows']) == result['ticks'] == 240


@pytest.mark.parametrize('fault', ('first', 'middle', 'terminal_only', 'stage', 'stage_clock',
                                 'terminated', 'early_truncated', 'no_completion', 'nan', 'force_scale'))
def test_bad_or_incomplete_telemetry_cannot_be_summarized(tmp_path, fault):
    rows = telemetry()
    if fault == 'first':
        rows.pop(0)
    elif fault == 'middle':
        rows.pop(3)
    elif fault == 'terminal_only':
        rows = rows[-1:]
    elif fault == 'stage':
        rows[1]['stage'] = 'right'
    elif fault == 'stage_clock':
        rows[1]['stage_time_s'] += .01
    elif fault == 'terminated':
        rows[-1]['terminated'] = 1
    elif fault == 'early_truncated':
        rows[1].update(truncated=1, termination_reason='time_limit')
    elif fault == 'no_completion':
        rows[-1].update(truncated=0, termination_reason='ongoing')
    elif fault == 'nan':
        rows[1]['yaw_integral_nm'] = float('nan')
    else:
        rows[1]['residual_vertical_n'] = 0.
    path = tmp_path/'telemetry.csv'
    put_csv(path, rows)
    with pytest.raises(ValueError):
        summarize(path)


@pytest.fixture
def inputs(tmp_path):
    run, audit = tmp_path/'run', tmp_path/'audit'
    config = {'schema': tool.pairs.PROTOCOL_SCHEMA, 'smoke_only': False,
              'training_seeds': [19000], 'checkpoint_budgets': [1, 2],
              'evaluation_cases': {
                  'development': [{'seed': seed, 'duration_s': .08} for seed in (17, 29)],
                  'holdout': [{'seed': 617, 'duration_s': duration} for duration in (.08, .1)]},
              'observation_schema': 'd1-continuous-sensor-command45-v1',
              'control_schema': 'd1-lqr-vmc-sensor-world-slopes-yaw-pi-v2',
              'action_schema': 'd1-terrain-residual-quarter-v1',
              'residual_force_scale_n': [11.25, 20.], 'stages_45_second_template': STAGES}
    put_json(run/'protocol.json', config)
    manifest = {'protocol.json': tool.pairs.sha((run/'protocol.json').read_bytes())}
    metrics = []
    for name, plan in tool.pairs.expected_matrix(config).items():
        metrics.append({**{key: value for key, value in plan.items() if key != 'expected_duration_s'},
                        'completed': True, 'quality_pass': False, 'duration_s': plan['expected_duration_s'],
                        **dict.fromkeys(tool.pairs.FIELDS, .1)})
        if plan['split'] == 'development' or plan['case'].startswith(('seed17_', 'seed29_')):
            continue
        relative = f'rollouts/{name}/protocol.json'
        metadata = None if plan['mode'] == 'zero' else {'training_seed': 19000, 'num_timesteps': 2}
        put_json(run/relative, {'mode': plan['mode'], 'seed': 617, 'control_dt_s': .01,
                               'config': {'duration_s': plan['expected_duration_s'], 'yaw_controller': 'pi'},
                               'checkpoint_metadata': metadata,
                               **{key: config[key] for key in ('observation_schema', 'control_schema', 'action_schema')}})
        manifest[relative] = tool.pairs.sha((run/relative).read_bytes())
        relative = f'rollouts/{name}/telemetry.csv'
        put_csv(run/relative, telemetry(plan['expected_duration_s']))
        manifest[relative] = tool.pairs.sha((run/relative).read_bytes())
    put_json(run/'manifest.json', {'sha256': manifest})
    put_csv(audit/'metrics_recomputed.csv', metrics)
    put_json(audit/'independent_audit.json', {
        'schema': tool.pairs.AUDIT_SCHEMA,
        'input_manifest_sha256': tool.pairs.sha((run/'manifest.json').read_bytes()),
        'complete_case_inventory_matches_protocol': True})
    refresh_audit(audit)
    return run, audit


def refresh_audit(audit):
    put_json(audit/'manifest.json', {'sha256': {name: tool.pairs.sha((audit/name).read_bytes())
             for name in ('metrics_recomputed.csv', 'independent_audit.json')}})


def test_bound_complete_pairs_stream_each_baseline_once_without_reading_npz(inputs, monkeypatch):
    run, audit = inputs
    original = Path.read_bytes

    def small_only(path):
        assert path.suffix not in ('.csv', '.npz', '.zip') or path.name == 'metrics_recomputed.csv'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', small_only)
    report = tool.diagnose(run, audit)
    assert report['status'] == 'DIAGNOSED' and report['holdout_pair_count'] == 2
    assert report['unique_case_count'] == 4 and report['total_control_ticks'] == 36
    assert len(report['raw_inputs']) == 8
    assert [row['control_ticks'] for row in report['descriptive_index']] == [18, 18]
    assert report['descriptive_index'][1]['vertical_action_at_clip'] == {
        'ticks': 18, 'denominator_ticks': 18}
    assert report['physics_steps'] == report['models_loaded'] == report['gui_starts'] == 0


def test_incomplete_or_missing_pair_stops_all_diagnosis_instead_of_survivor_analysis(inputs):
    run, audit = inputs
    path = audit/'metrics_recomputed.csv'
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    rows = [row for row in rows if not row['rollout'].startswith('holdout_train19000_seed617_0.1s')]
    put_csv(path, rows)
    refresh_audit(audit)
    report = tool.diagnose(run, audit)
    assert report['status'] == 'INCOMPARABLE'
    assert report['cases'] == [] and report['raw_inputs'] == {}


@pytest.mark.parametrize('fault', ('csv', 'protocol', 'metrics'))
def test_wrong_identity_is_rejected_before_report(inputs, fault):
    run, audit = inputs
    if fault == 'metrics':
        path = audit/'metrics_recomputed.csv'
    else:
        path = run/'rollouts/holdout_train19000_seed617_0.08s'/('telemetry.csv' if fault == 'csv' else 'protocol.json')
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        tool.diagnose(run, audit)


def test_cli_needs_no_site_packages_and_rejects_overwrite_or_input_output(inputs, tmp_path):
    run, audit = inputs
    output = tmp_path/'diagnostic.json'
    result = subprocess.run([sys.executable, '-S', '-B', str(SCRIPT), '--run', str(run),
                             '--audit', str(audit), '--output', str(output)], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['status'] == 'DIAGNOSED'
    before = output.read_bytes()
    assert tool.main(['--run', str(run), '--audit', str(audit), '--output', str(output)]) == 2
    assert output.read_bytes() == before
    assert tool.main(['--run', str(run), '--audit', str(audit), '--output', str(run/'new.json')]) == 2


def test_cli_incomparable_is_reported_with_exit_one(inputs, tmp_path):
    run, audit = inputs
    path = audit/'metrics_recomputed.csv'
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    next(row for row in rows if row['split'] == 'holdout')['completed'] = 'False'
    put_csv(path, rows)
    refresh_audit(audit)
    output = tmp_path/'incomplete.json'
    assert tool.main(['--run', str(run), '--audit', str(audit), '--output', str(output)]) == 1
    assert json.loads(output.read_text())['cases'] == []
