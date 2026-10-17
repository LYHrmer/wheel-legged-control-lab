"""Small fake receipts and path/AST checks, with no worker or reader import."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_interaction33', ROOT/'scripts/run_b22_interaction33.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def baseline(tmp_path):
    # A saved report is only a small JSON fixture; no trace/model is loaded.
    report = runner.c.read_json(ROOT/'results/d1_b22_gui_true15mm_20260930/continuation26/independent_keyboard_gui_26.json')
    run = tmp_path/'keyboard_gui'
    session, worker = {'bytes': 1, 'sha256': 'a'*64}, {'bytes': 2, 'sha256': 'b'*64}
    report.update(run=str(run), session_identity=session, worker_receipt_identity=worker)
    kwargs = {'run': run,
              'source_inputs': {str(ROOT/name): runner.c.identity(ROOT/name)
                                for name in (runner.READER, runner.BASE_READER)}, 'returncode': 0,
              'session_id': session, 'worker_id': worker}
    return report, kwargs


def test_plan_preserves_the_exact_c26_events_and_global_budget():
    value = runner.plan()
    assert value['status'] == 'PLAN_ONLY' and value['execution_started'] is False
    assert value['model_loads'] == value['physics_steps'] == 0
    assert value['limits']['controls'] == 1200 and value['limits']['normal_native'] == 6000
    assert value['limits']['compiler_native'] == 2 and value['limits']['total_wall_s'] == 900
    assert value['reset_refunds_budget'] is value['retry_permitted'] is False
    events = value['x11_events']
    assert len(events) == 27
    assert [event['tick'] for event in events] == [100, 180, 200, 230, 250, 280, 350,
        400, 405, 430, 450, 470, 500, 510, 520, 590, 610, 650, 660, 720, 725, 760,
        800, 850, 855, 860, 1100]
    assert events[13] == {'key': 'w', 'operation': 'key_up', 'tick': 510}
    assert events[-1] == {'key': 'Escape', 'operation': 'key_down', 'tick': 1100}
    actual = value['execution_spec_without_output']
    assert actual['execution_contract_id'] == runner.CONTRACT
    assert (actual['soft_s'], actual['close_s'], actual['hard_s']) == (95, 115, 120)
    assert actual['seed'] == 231002 and actual['require_event_driver'] is True
    assert actual['test_input_pause']['start_control'] == 530
    assert actual['test_input_pause']['duration_s'] == .35
    assert actual['model_limits'] == runner.c.model_limits('B', 1200, actual['model_limits'])


def test_new_request_changes_only_output_and_uses_owned_x11(tmp_path):
    config = runner.c.read_json(ROOT/runner.c.MANIFEST)
    release_path = tmp_path/'release.json'
    runner.c.write_json(release_path, {'synthetic': True})
    original, events = runner.template()
    argv = runner.make_request(tmp_path, config, {'inputs': {'synthetic': {}}}, release_path)
    request = runner.c.read_json(tmp_path/'launch/request.json')
    go = runner.c.read_json(tmp_path/'launch/source_go.json')
    expected = {**original, 'output_directory': str(tmp_path/'keyboard_gui')}
    assert request['spec'] == go['arms']['keyboard_gui'] == expected
    assert request['x11_events'] == go['x11_events_by_arm']['keyboard_gui'] == events
    assert go['worker'] == str(ROOT/runner.WORKER)
    assert go['runtime_environment'] == config['runtime_environment']
    assert go['shared_session'] == config['shared_session']
    assert go['inputs'][str(release_path)] == runner.c.identity(release_path)
    assert argv == ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'],
                    '--request', str(tmp_path/'launch/request.json'), '--isolated-x11', str(runner.ISOLATION)]
    assert not (tmp_path/'keyboard_gui').exists()


def test_reader_uses_the_original_saved_only_cli_without_adapting_math(tmp_path):
    assert runner.readback_argv(tmp_path) == ['rtk', 'proxy', '/usr/bin/python3', '-B',
        str(ROOT/runner.READER), '--run', str(tmp_path/'keyboard_gui'),
        '--output', str(tmp_path/'readback.json')]
    tree = ast.parse((ROOT/runner.READER).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    calls = {ast.unparse(node.func) for node in ast.walk(main) if isinstance(node, ast.Call)}
    assert {'base.read_run_details', 'verify_deferred'} <= calls


@pytest.mark.parametrize('change', [None, 'display', 'command', 'output', 'auth', 'cleanup', 'session'])
def test_owned_x11_receipt_must_belong_to_the_same_session(tmp_path, change):
    config = {'host': 'frozen-host'}
    log = tmp_path/'keyboard_gui.x11'
    receipt = {'display': ':183', 'child_output': str(tmp_path/'keyboard_gui'), 'logs': str(log),
               'command': ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'],
                           '--request', str(tmp_path/'launch/request.json'), '--owned-worker-host'],
               'status': 'passed', 'child_returncode': 0, 'auth_cookie_logged': False,
               'auth_removed': True, 'owned_server_stopped': True}
    session = {'runtime_environment': {'DISPLAY': ':183', 'XAUTHORITY': str(log/'Xauthority')}}
    if change == 'display':
        receipt['display'] = ':1'
    elif change == 'command':
        receipt['command'] = []
    elif change == 'output':
        receipt['child_output'] = 'old-history'
    elif change == 'auth':
        receipt['auth_removed'] = False
    elif change == 'cleanup':
        receipt['owned_server_stopped'] = False
    elif change == 'session':
        session['runtime_environment']['DISPLAY'] = ':1'
    if change is None:
        runner.validate_isolation(receipt, session, tmp_path, config)
    else:
        with pytest.raises(ValueError, match='X11 server/session/cleanup'):
            runner.validate_isolation(receipt, session, tmp_path, config)


def test_actual_saved_report_satisfies_new_receipt_gate(baseline):
    report, kwargs = baseline
    counts, qualified = runner.validate_readback(report, **kwargs)
    assert qualified is True
    assert counts == {'controls': 1102, 'normal_native': 5510, 'compiler_native': 2}


@pytest.mark.parametrize('field,value', [
    ('run', 'old-history'), ('record_valid', False), ('source_and_actual_origin_verified', False),
    ('checkpoint_sha256', '0'*64), ('reader_model_calls', 1), ('reader_physics_calls', 1),
    ('schema', 'not-the-frozen-reader'), ('execution_contract_id', 'C23_B22_GUI_v1'),
    ('reader_source_identity', {}), ('session_identity', {}), ('worker_receipt_identity', {}),
    ('base_reader_source_identity', {}), ('completed_controls', 1201),
    ('normal_native_returns', 6001), ('compiler_native_returns', 3), ('actual_episode_count', 3),
])
def test_wrong_identity_or_budget_is_not_completed(baseline, field, value):
    report, kwargs = baseline
    report[field] = value
    with pytest.raises(ValueError):
        runner.validate_readback(report, **kwargs)


def test_callback_success_cannot_replace_actual_prepared_input_consumption(baseline):
    report, kwargs = baseline
    assert report['real_events']['passed'] is True
    report['input']['controls_with_exact_preparation'] -= 1
    with pytest.raises(ValueError, match='input consumption'):
        runner.validate_readback(report, **kwargs)


def test_reset_cannot_refund_or_multiply_global_budget(baseline):
    report, kwargs = baseline
    report['input']['actual_reset_count'] = 2
    with pytest.raises(ValueError, match='reset/global'):
        runner.validate_readback(report, **kwargs)


@pytest.mark.parametrize('key', ['learn', 'train', 'save', 'forward', 'backward', 'predict_values'])
def test_training_or_extra_model_api_is_rejected(baseline, key):
    report, kwargs = baseline
    report['model_calls']['counts'][key] = {'attempted': 1, 'returned': 1}
    with pytest.raises(ValueError, match='model/probe budget'):
        runner.validate_readback(report, **kwargs)


def test_extra_prediction_or_actor_rows_are_rejected(baseline):
    report, kwargs = baseline
    report['model_calls']['actor_rows'] += 1
    with pytest.raises(ValueError, match='model/probe budget'):
        runner.validate_readback(report, **kwargs)


@pytest.mark.parametrize('key', ['real_events', 'render', 'last100_stop', 'deferred_archive', 'owner_join'])
def test_a_failed_gate_cannot_be_hidden_behind_qualification_true(baseline, key):
    report, kwargs = baseline
    report[key]['passed'] = False
    with pytest.raises(ValueError, match='mandatory gates'):
        runner.validate_readback(report, **kwargs)


def test_valid_performance_failure_is_retained_without_claiming_qualification(baseline):
    report, kwargs = baseline
    report['render']['passed'] = False
    report['arm_qualification_passed'] = report['execution_prerequisite_passed'] = False
    kwargs['returncode'] = 2
    counts, qualified = runner.validate_readback(report, **kwargs)
    assert qualified is False and counts['controls'] == 1102
    kwargs['returncode'] = 0
    with pytest.raises(ValueError, match='exit/qualification'):
        runner.validate_readback(report, **kwargs)


def test_default_entry_stays_cold_and_does_not_create_output(tmp_path):
    output = tmp_path/'not-created'
    code = ('import importlib.util,sys;'
            's=importlib.util.spec_from_file_location("cold33","scripts/run_b22_interaction33.py");'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'assert not any(x in sys.modules for x in '
            '("torch","mujoco","numpy","glfw","stable_baselines3","reader26"));'
            f'm.main(["--output",{str(output)!r}])')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'PLAN_ONLY'
    assert not output.exists()


@pytest.mark.parametrize('resource', ['RAM', 'disk'])
def test_admission_rejects_before_output_or_launch(monkeypatch, resource):
    output = ROOT/'results/b22_interaction33_pytest_never_create'
    assert not output.exists()
    monkeypatch.setattr(runner, 'REGISTERED_ROOT', ROOT)
    monkeypatch.setattr(runner, 'source_identity', dict)
    monkeypatch.setattr(runner.common, 'memory', lambda: (5 if resource == 'RAM' else 7) << 30)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1 << 30))
    monkeypatch.setattr(runner.common, 'supervise', lambda *a, **kw: pytest.fail('launched before resource admission'))
    with pytest.raises(ValueError, match='NOT_READY'):
        runner.execute(output)
    assert not output.exists()


def test_execute_is_restricted_to_the_registered_machine_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='new direct'):
        runner.execute(tmp_path/'results/b22_interaction33_other_machine')


def test_existing_output_is_never_reused(tmp_path):
    with pytest.raises(ValueError, match='new direct'):
        runner.execute(tmp_path)


def test_pinned_historical_input_and_helpers_cannot_drift(monkeypatch):
    monkeypatch.setattr(runner, 'GO_ID', {'bytes': 1, 'sha256': '0'*64})
    with pytest.raises(ValueError, match='event source changed'):
        runner.template()
    monkeypatch.setattr(runner, 'HELPERS', {'scripts/run_b22_suite31.py': '0'*64})
    with pytest.raises(ValueError, match='helper changed'):
        runner._shared()


def test_new_source_closure_includes_old_events_isolation_and_supervisor():
    config = runner.c.read_json(ROOT/runner.c.MANIFEST)
    frozen = runner.inputs(config, {'authored': {'synthetic-new-source': {}}})
    assert frozen['synthetic-new-source'] == {}
    assert frozen[str(ROOT/runner.GO)] == runner.GO_ID
    assert all(frozen[key] == value for key, value in runner.EXTRAS.items())
    assert all(frozen[str(ROOT/name)]['sha256'] == digest for name, digest in runner.HELPERS.items())


def test_source_change_rejects_readback_without_loading_any_runtime(tmp_path):
    path = tmp_path/'source.py'
    path.write_text('first')
    frozen = {str(path): runner.c.identity(path)}
    runner.validate_sources(frozen)
    path.write_text('second')
    with pytest.raises(ValueError, match='input changed'):
        runner.validate_sources(frozen)


def test_subreaper_precedes_both_real_launch_sites():
    tree = ast.parse((ROOT/'scripts/run_b22_interaction33.py').read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    calls = [node for node in ast.walk(owner) if isinstance(node, ast.Call)]
    reaper = next(node for node in calls if ast.unparse(node.func) == 'host.subreaper')
    launches = [node for node in calls if ast.unparse(node.func) == 'common.supervise']
    assert len(launches) == 2 and all(reaper.lineno < node.lineno for node in launches)
    assert not any(ast.unparse(node.func) == 'subprocess.Popen' for node in calls)


def test_exact_template_is_not_mutated_by_callers():
    original, events = runner.template()
    saved = copy.deepcopy((original, events))
    original['control_limit'] = 6000
    events[0]['tick'] = 1
    assert runner.template() == saved
