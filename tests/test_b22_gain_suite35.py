"""Synthetic C35 owners, whole metadata seams, original numeric scoring; zero physics."""
from __future__ import annotations

import ast
import copy
import importlib.machinery
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = module('test_gain_suite35_runner', 'scripts/run_b22_gain_suite35.py')
reader = module('test_gain_suite35_reader', 'scripts/read_b22_gain_suite35.py')
worker = module('test_gain_suite35_worker', 'scripts/run_b22_worker35.py')


def test_fresh_whole_matrix_budget_windows_and_resources():
    stages = runner.stages()
    assert len(stages) == 15 and stages[0]['name'] == 'step_pair'
    assert sum(s['controls'] for s in stages) == 25200
    assert sum(s['compiler_native'] for s in stages) == 30
    assert sum(5*s['controls'] for s in stages) == 126000
    assert len({s['name'] for s in stages}) == 15
    assert runner.LIMITS['total_wall_s'] == 1800
    assert runner.LIMITS['start_effective_bytes'] == 6 << 30
    assert runner.LIMITS['reserve_effective_bytes'] == 4 << 30
    assert runner.LIMITS['required_minimum_nice'] == 10
    assert all(s['gain35'] == .5 for s in stages)
    assert [(s['controls'], s['release_tick'], s['hold']) for s in stages if s.get('terrain') == 'ramp'] == [(1800, 1355, [600, 1000])]*2
    assert len([s for s in stages if s.get('mirror')]) == 2


def trigger_fixture():
    baseline = {'status': 'COMPLETE', 'RL_contribution_passed': False,
                'original_six_contribution': {'prerequisites': {'passed': True}}}
    probe = {'status': 'COMPLETE', 'all_four_full_horizons': True, 'all_four_tasks_passed': True,
             'all_two_profile_SSE_and_cost_lower': True, 'full_window_ratios_available': True,
             'pairs': [{'profile': profile, 'SSE_gain0p5_over_gain1': .9, 'cost_gain0p5_over_gain1': .95}
                       for profile in ('flat_1p6', 'flat_1p2_yaw')]}
    return baseline, probe


@pytest.mark.parametrize('change', [None, 'partial', 'prerequisite', 'contribution', 'horizon', 'task', 'SSE', 'cost', 'missing', 'order'])
def test_trigger_cannot_turn_local_or_partial_evidence_into_full_adoption(change):
    baseline, probe = trigger_fixture()
    if change == 'partial':
        baseline['status'] = 'INCOMPLETE'
    elif change == 'prerequisite':
        baseline['original_six_contribution']['prerequisites']['passed'] = False
    elif change == 'contribution':
        baseline['RL_contribution_passed'] = True
    elif change == 'horizon':
        probe['all_four_full_horizons'] = False
    elif change == 'task':
        probe['all_four_tasks_passed'] = False
    elif change in ('SSE', 'cost'):
        probe['pairs'][0][change+'_gain0p5_over_gain1'] = 1.
    elif change == 'missing':
        probe['pairs'].pop()
    elif change == 'order':
        probe['pairs'].reverse()
    if change is None:
        assert runner.trigger_decision(baseline, probe)['local_two_profile_improvement'] is True
    else:
        with pytest.raises(ValueError):
            runner.trigger_decision(baseline, probe)


def find_function(source, name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_full_script_owner_ast_changes_only_admission_and_versioned_gain_keys():
    old = ast.parse((ROOT/worker.SCRIPT).read_text())
    actual = ast.parse(worker.derived_sources()['script'])
    old_pre = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
    new_pre = next(n for n in actual.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
    new_pre.body[0] = copy.deepcopy(old_pre.body[0])
    for node in ast.walk(actual):
        if isinstance(node, ast.Constant) and node.value == 'gain35':
            node.value = 'gain34'
    assert ast.dump(actual, include_attributes=False) == ast.dump(old, include_attributes=False)


def test_full_step_owner_ast_preserves_compile_native_archive_cleanup_and_model_ledger():
    old = ast.parse((ROOT/(worker.STEP+'worker24.py')).read_text())
    new = ast.parse(worker.derived_sources()['step_worker'])
    original_main = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    actual_main = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    old_try = next(n for n in original_main.body if isinstance(n, ast.Try))
    new_try = next(n for n in actual_main.body if isinstance(n, ast.Try))
    new_try.body[0] = copy.deepcopy(old_try.body[0])
    for node in ast.walk(new):
        if isinstance(node, ast.Constant) and node.value == worker.CONTRACT:
            node.value = 'C24_true_15mm_B22_v1'
        elif isinstance(node, ast.Constant) and node.value == 'd1-true15mm-worker-35-v1':
            node.value = 'd1-true15mm-worker-24-v1'
    class RestoreEvaluator(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node) == 'evaluate = _evaluate35':
                return ast.ImportFrom(module='eval24', names=[ast.alias(name='evaluate')], level=0)
            return self.generic_visit(node)
    assert ast.dump(RestoreEvaluator().visit(new), include_attributes=False) == ast.dump(old, include_attributes=False)


def test_step_record_and_evaluator_ast_only_change_network_scaling_ledger_and_contract():
    source = worker.derived_sources()
    old_record = ast.parse((ROOT/(worker.STEP+'heldout24.py')).read_text())
    new_record = ast.parse(source['step_record'])
    class RestoreRecord(ast.NodeTransformer):
        def visit_Assign(self, node):
            text = ast.unparse(node)
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == '_numeric_actor_action':
                return ast.parse('action, predicted = _numeric_actor_action(score_actor, policy, observation)').body[0]
            if text.startswith('network_output16 = np.asarray') or text == 'action = network_output16 * np.float32(0.5)':
                return None
            return self.generic_visit(node)
        def visit_Dict(self, node):
            keep = [i for i, key in enumerate(node.keys) if not (isinstance(key, ast.Constant)
                    and key.value in ('network_output16', 'gain35', 'raw_action16', 'action16'))]
            node.keys, node.values = [node.keys[i] for i in keep], [node.values[i] for i in keep]
            return self.generic_visit(node)
    assert ast.dump(RestoreRecord().visit(new_record), include_attributes=False) == ast.dump(old_record, include_attributes=False)
    expected = (ROOT/(worker.STEP+'eval24.py')).read_text().replace('from heldout24 import record_case', 'record_case = _record_case35')
    expected = expected.replace('execution_contract_id=CONTRACT_ID', 'execution_contract_id='+repr(worker.CONTRACT))
    assert ast.dump(ast.parse(source['step_eval']), include_attributes=False) == ast.dump(ast.parse(expected), include_attributes=False)


@pytest.mark.parametrize('kind', ['script', 'step_record'])
@pytest.mark.parametrize('actor', ['B', 'zero'])
@pytest.mark.parametrize('edge', [False, True])
def test_actual_action_seams_reuse_numeric_actor_and_zero_never_predicts(kind, actor, edge):
    source = (ROOT/'runtime/d1_b22/research/run_rl16_training_08.py').read_text()
    numeric = find_function(source, '_numeric_actor_action')
    namespace = {'Any': object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[numeric], type_ignores=[])), '<frozen-numeric-action>', 'exec'), namespace)  # noqa: S102 -- pure fixed numeric actor function
    original = np.linspace(-.9, .9, 16, dtype=np.float32)
    if edge:
        tiny = np.nextafter(np.float32(0.), np.float32(1.))
        original = np.asarray([-1., 1., -0., 0., tiny, -tiny, 2*tiny, -2*tiny]*2, dtype=np.float32)
    initial = original.copy()
    calls = []
    policy = SimpleNamespace(predict=lambda obs, deterministic: (calls.append(deterministic) or original, None)) if actor == 'B' else None
    owner = find_function(worker.derived_sources()[kind], 'owner' if kind == 'script' else 'record_case')
    statements = sorted((n for n in ast.walk(owner) if isinstance(n, ast.Assign)), key=lambda n: n.lineno)
    first = next(n for n in statements if isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == '_numeric_actor_action')
    selected = [n for n in statements if first.lineno <= n.lineno <= first.end_lineno+2]
    assert len(selected) == 3
    namespace.update(np=np, policy=policy, observation=np.zeros(99, np.float32), score_actor='final_policy' if actor == 'B' else 'zero', session={'gain35': .5})
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), '<actual-C35-action-seam>', 'exec'), namespace)  # noqa: S102 -- actual generated action block only
    expected = initial if actor == 'B' else np.zeros(16, np.float32)
    assert np.array_equal(namespace['action'], expected*np.float32(.5))
    assert namespace['action'].dtype == np.float32 and namespace['predicted'] is (actor == 'B')
    assert calls == ([True] if actor == 'B' else []) and np.array_equal(initial, original)


@pytest.mark.parametrize('actor', ['B', 'zero'])
@pytest.mark.parametrize('change', [None, 'gain', 'network', 'raw', 'prediction', 'nan', 'shape'])
def test_reader_requires_actual_controller_input_gain_and_prediction_ledger(actor, change):
    network = np.linspace(-.8, .8, 16, dtype=np.float32) if actor == 'B' else np.zeros(16, np.float32)
    action = (network*np.float32(.5)).tolist()
    row = {'network_output16': network.tolist(), 'gain35': .5, 'action16': action,
           'raw_action16': action, 'policy_input_action': action, 'policy_predict_called': actor == 'B'}
    if change == 'gain':row['gain35'] = 1.
    elif change == 'network':row['network_output16'][0] += .1
    elif change == 'raw':row['raw_action16'] = [1.]*16
    elif change == 'prediction':row['policy_predict_called'] = actor != 'B'
    elif change == 'nan':row['network_output16'][0] = float('nan')
    elif change == 'shape':row['network_output16'].pop()
    if change is None:assert reader.check_gain_row(row, actor) == 1
    else:
        with pytest.raises(ValueError):reader.check_gain_row(row, actor)


@pytest.mark.parametrize('index', [0, 1, 2, 11, 12, 13, 14])
def test_requests_are_real_C35_owners_and_keep_original_model_api_caps(tmp_path, index):
    config = runner.c.read_json(ROOT/runner.c.MANIFEST)
    release_path = tmp_path/'release.json'
    runner.c.write_json(release_path, {'synthetic': True})
    stage = runner.stages()[index]
    argv = runner.request(stage, tmp_path, {'inputs': {}}, config, release_path)
    request = runner.c.read_json(tmp_path/(stage['name']+'_launch/request.json'))
    go = runner.c.read_json(request['source_go_path'])
    spec = request['spec']
    assert go['worker'] == str(ROOT/'scripts/run_b22_worker35.py')
    assert spec['execution_contract_id'] == worker.CONTRACT and spec['gain35'] == .5
    assert spec['control_limit'] == stage['controls'] and not Path(spec['output_directory']).exists()
    assert spec['model_limits']['predict'] == (1201 if stage['kind'] == 'step' else stage['controls']+1 if stage['actor'] == 'B' else 0)
    assert spec['model_limits']['load'] == (1 if stage['kind'] == 'step' or stage['actor'] == 'B' else 0)
    assert argv[4] == config['host']


def test_dispatcher_defers_step_imports_until_old_owner_requests_evaluation(tmp_path, monkeypatch):
    session = tmp_path/'session.json'
    session.write_text(json.dumps({'execution_contract_id': worker.CONTRACT, 'gain35': .5,
                                   'mode': 'eval', 'output_directory': str(tmp_path)}))
    marker = tmp_path/'order.txt'
    sources = {'script': '', 'step_record': f'from pathlib import Path\nPath({str(marker)!r}).open("a").write("record\\n")\ndef record_case(*a):return "recorded"\n',
               'step_eval': 'def evaluate(*args):return _record_case35(*args)\n',
               'step_worker': f'from pathlib import Path\np=Path({str(marker)!r})\nassert not p.exists()\np.write_text("worker-preflight-engine\\n")\nassert _evaluate35(1)=="recorded"\n'}
    monkeypatch.setattr(worker, 'derived_sources', lambda: sources)
    worker.main(['--session', str(session), '--output', str(tmp_path)])
    assert marker.read_text() == 'worker-preflight-engine\nrecord\n'
    assert json.loads((tmp_path/'derived_sources35.json').read_text())['execution_contract_id'] == runner.CONTRACT


def test_default_import_and_plan_are_cold_and_create_nothing(tmp_path):
    output = tmp_path/'must-not-exist'
    code = ('import importlib.util,sys;s=importlib.util.spec_from_file_location("cold35","scripts/run_b22_gain_suite35.py");'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'assert not any(x in sys.modules for x in ("numpy","torch","mujoco","onnxruntime","glfw","stable_baselines3","gymnasium"));'
            f'm.main(["--output",{str(output)!r}])')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['execution_started'] is False and not output.exists()


@pytest.mark.parametrize('change', [None, 'contract', 'schema', 'count', 'compiler', 'ledger', 'reader', 'release'])
def test_outer_stage_receipt_gate_before_next_session(change):
    stage = runner.stages()[2]
    rid, did = {'sha256': 'release'}, {'sha256': 'reader'}
    value = {'schema': 'b22-gain-suite35-stage-readback-v1', 'status': 'COMPLETE', 'stage': stage['name'],
             'execution_contract_id': runner.CONTRACT, 'gain35': .5, 'release_identity': rid, 'reader_identity': did,
             'actual_counts': {'controls': 1600, 'normal_native': 8000, 'compiler_native': 2}, 'gain_ledger_rows': 1600}
    if change == 'contract':value['execution_contract_id'] = 'C31_B22_fixed_suite_v1'
    elif change == 'schema':value['schema'] = 'b22-gain34-stage-readback-v1'
    elif change == 'count':value['actual_counts']['normal_native'] = 7999
    elif change == 'compiler':value['actual_counts']['compiler_native'] = 1
    elif change == 'ledger':value['gain_ledger_rows'] = 1599
    elif change in ('reader', 'release'):value[change+'_identity'] = {}
    if change is None:assert runner.stage_counts(stage, value, rid, did) == value['actual_counts']
    else:
        with pytest.raises(ValueError):runner.stage_counts(stage, value, rid, did)


@pytest.mark.parametrize('change', [None, 'missing', 'foreign', 'native', 'compiler'])
def test_partial_counts_are_owned_facts_and_unknown_is_not_zero(tmp_path, change):
    stage = runner.stages()[2]
    session = {'execution_contract_id': runner.CONTRACT, 'gain35': .5,
               'output_directory': str(tmp_path), 'source_hashes': {'synthetic': {'sha256': 'bound'}}}
    runner.c.write_json(tmp_path/'session.json', session)
    receipt = {'execution_contract_id': runner.CONTRACT, 'session_identity': runner.c.identity(tmp_path/'session.json'),
               'result': {'python': {'control_completed': 10, 'native_returned': 52}, 'C_final': {'construction_returns': 2}}}
    if change == 'foreign':receipt['session_identity'] = {}
    elif change == 'native':receipt['result']['python']['native_returned'] = 60
    elif change == 'compiler':receipt['result']['C_final']['construction_returns'] = 3
    if change != 'missing':runner.c.write_json(tmp_path/'worker_receipt.json', receipt)
    result = runner.known_partial(stage, tmp_path, {'inputs': session['source_hashes']})
    assert result == ({'controls': 10, 'normal_native': 52, 'compiler_native': 2} if change is None else None)


def test_subreaper_precedes_every_launch_and_actual_nice_gate_is_enforced():
    source = (ROOT/'scripts/run_b22_gain_suite35.py').read_text()
    owner = find_function(source, 'execute')
    calls = [n for n in ast.walk(owner) if isinstance(n, ast.Call)]
    reaper = next(n for n in calls if ast.unparse(n.func) == 'host.subreaper')
    launches = [n for n in calls if ast.unparse(n.func) == 'common.supervise']
    assert len(launches) == 3 and all(reaper.lineno < n.lineno for n in launches)
    assert 'os.getpriority(os.PRIO_PROCESS, 0) >= 10' in source
    assert not any(ast.unparse(n.func) == 'subprocess.Popen' for n in calls)


@pytest.fixture(scope='module')
def numeric_cases():
    """Reuse existing small canonical fixtures and the actual unchanged C18 scorer."""
    fixtures = module('gain_suite35_fixture31', 'tests/test_b22_suite31.py')
    before = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT/'runtime/d1_b22/research/course_impl08'))
        scorer = module('gain_suite35_score18', 'runtime/d1_b22/research/continuation18/score18.py')
        reports = []
        for stage in runner.stages():
            value = {'schema': 'b22-gain-suite35-stage-readback-v1', 'status': 'COMPLETE', 'stage': stage['name'],
                     'execution_contract_id': runner.CONTRACT, 'gain35': .5,
                     'release_identity': {'synthetic': 'release'}, 'reader_identity': {'synthetic': 'reader'},
                     'actual_counts': {'controls': stage['controls'], 'normal_native': 5*stage['controls'], 'compiler_native': 2},
                     'gain_ledger_rows': stage['controls'],
                     'cases': [{'actor': 'zero', 'task_qualified': True}, {'actor': 'B', 'task_qualified': True}]}
            if stage['kind'] == 'script':
                source = fixtures.canonical(stage, 'zero' if stage['actor'] == 'zero' else 'final_policy',
                                            error=.01 if stage['actor'] == 'zero' else .005)
                source['positive_wheel_load_native_counts_by_family']['bump'] = 1
                if stage['actor'] == 'B':source['motor_torque_nm'] *= .9
                value['cases'] = [scorer.score_case(source, expected_seed=stage['seed'], mirror=stage['mirror'])]
                value['pairing_identity'] = {'five_arrays': 'same', 'complete_reset': 'same'}
            reports.append(value)
        prior = runner.pinned('gain_suite35_test_aggregate31', ROOT/'scripts/read_b22_suite31.py')
        baseline = prior.aggregate(copy.deepcopy(reports), scorer.score_pairs)
        yield fixtures, scorer, reports, baseline
    finally:
        sys.path[:] = before


def test_actual_original_six_gate_and_all_seven_tasks_are_retained(numeric_cases):
    _, scorer, reports, baseline = numeric_cases
    value = reader.aggregate(copy.deepcopy(reports), scorer.score_pairs, baseline)
    assert value['RL_contribution_passed'] is True
    assert value['finite_suite_task_qualification_passed'] is True and value['no_task_regression'] is True
    assert len(value['script_cases']) == 14 and len(value['mirror_scores_separate']) == 2
    assert len(value['real_15mm_scores_separate']) == 2 and value['gain35'] == .5
    assert value['adoption_decision'] == 'NOT_AUTOMATIC' and value['default_controller_or_model_changed'] is False


def test_short_policy_low_total_error_cannot_pass_original_contribution_gate(numeric_cases):
    fixtures, scorer, original, baseline = numeric_cases
    reports = copy.deepcopy(original)
    stage = runner.stages()[2]
    source = fixtures.canonical(stage, 'final_policy', error=.00001, n=100, failure=True)
    item = reports[2]
    item['actual_counts'].update(controls=100, normal_native=500)
    item['gain_ledger_rows'] = 100
    item['cases'] = [scorer.score_case(source, expected_seed=stage['seed'], mirror=stage['mirror'])]
    value = reader.aggregate(reports, scorer.score_pairs, baseline)
    assert value['RL_contribution_passed'] is False and value['no_task_regression'] is False
    assert value['finite_suite_task_qualification_passed'] is False and value['task_regressions']


@pytest.mark.parametrize('change', ['mirror_task', 'step_task', 'pair', 'window', 'actor', 'seed', 'missing', 'order', 'pretend_full'])
def test_whole_matrix_errors_or_separate_task_regressions_remain_visible(numeric_cases, change):
    fixtures, scorer, original, baseline = numeric_cases
    reports = copy.deepcopy(original)
    if change == 'mirror_task':
        stage = runner.stages()[-1]
        source = fixtures.canonical(stage, 'final_policy', error=.00001, n=100, failure=True)
        reports[-1]['cases'] = [scorer.score_case(source, expected_seed=stage['seed'], mirror=True)]
        reports[-1]['actual_counts'].update(controls=100, normal_native=500)
        reports[-1]['gain_ledger_rows'] = 100
    elif change == 'step_task':reports[0]['cases'][1]['task_qualified'] = False
    elif change == 'pair':reports[2]['pairing_identity'] = {}
    elif change == 'window':reports[2]['cases'][0]['windows']['final'] = [100, 200]
    elif change == 'actor':reports[2]['cases'][0]['actor'] = 'zero'
    elif change == 'seed':reports[2]['cases'][0]['seed'] += 1
    elif change == 'missing':reports.pop()
    elif change == 'order':reports.reverse()
    elif change == 'pretend_full':
        reports[2]['actual_counts'].update(controls=100, normal_native=500)
        reports[2]['gain_ledger_rows'] = 100
        reports[2]['cases'][0]['completed_controls'] = 100
    if change in ('mirror_task', 'step_task'):
        value = reader.aggregate(reports, scorer.score_pairs, baseline)
        assert value['finite_suite_task_qualification_passed'] is False and value['no_task_regression'] is False
        assert value['RL_contribution_passed'] is True  # Separate task failure cannot be hidden by the original six.
    else:
        with pytest.raises(ValueError):reader.aggregate(reports, scorer.score_pairs, baseline)


@pytest.mark.parametrize('change', [None, 'old_contract', 'native'])
def test_entire_script_source_archive_and_native_metadata_function_accepts_C35(tmp_path, change):
    # The C34 helper creates only synthetic metadata and executes the actual
    # frozen archive/check_ledgers functions. Here every saved identity is
    # explicitly a newly authored C35 fixture, including its real entry path.
    fixtures = module('gain_suite35_metadata_fixture34', 'tests/test_b22_gain34.py')
    old_checker, args = fixtures.complete_metadata_fixture(tmp_path, .5)
    run, stage, release, base = args
    prior = SimpleNamespace(**old_checker.__globals__)
    old_worker = str(ROOT/'scripts/run_b22_worker34.py')
    new_worker = str(ROOT/'scripts/run_b22_worker35.py')
    def rewrite(value):
        if isinstance(value, dict):
            result = {(new_worker if k == old_worker else 'gain35' if k == 'gain34' else k): rewrite(v) for k, v in value.items()}
            if new_worker in result:result[new_worker] = runner.c.identity(new_worker)
            return result
        if isinstance(value, list):return [rewrite(v) for v in value]
        return (new_worker if value == old_worker else runner.CONTRACT if value == fixtures.runner.CONTRACT else value)
    paths = [p for p in tmp_path.rglob('*.json') if not p.name.endswith('.manifest.json')]
    documents = {path: rewrite(json.loads(path.read_text())) for path in paths}
    for path, value in documents.items():path.write_text(json.dumps(value))
    session_path = run/'session.json'
    session = documents[session_path]
    session['source_go_identity'] = runner.c.identity(session['source_go_path'])
    if change == 'old_contract':session['execution_contract_id'] = 'C31_B22_fixed_suite_v1'
    session_path.write_text(json.dumps(session))
    wpath = run/'worker_receipt.json'
    receipt = documents[wpath]
    receipt['session_identity'] = runner.c.identity(session_path)
    if change == 'native':receipt['result']['python']['native_returned'] -= 1
    wpath.write_text(json.dumps(receipt))
    ready = documents[run/'preflight_ready.json']
    ready['session_sha256'] = runner.c.identity(session_path)['sha256']
    (run/'preflight_ready.json').write_text(json.dumps(ready))
    for path in paths:
        manifest = path.with_name(path.name+'.manifest.json')
        if manifest.exists():
            manifest.write_text(json.dumps({'schema': 'd1-archive-transaction-13-v1',
                                           'payloads': [{'file': path.name, **runner.c.identity(path)}]}))
    release = rewrite(release)
    checker = reader.metadata_checker(prior)
    if change is None:
        session, receipt, construction, count = checker(run, stage, release, base)
        assert count == 1600 and session['execution_contract_id'] == receipt['execution_contract_id'] == runner.CONTRACT
        assert construction['proof']['passed'] is True
    else:
        with pytest.raises(ValueError):checker(run, stage, release, base)


@pytest.mark.parametrize('kind', ['script', 'step'])
@pytest.mark.parametrize('change', [None, 'old_contract', 'schema', 'native', 'warning'])
def test_C35_ending_and_count_gate_uses_versioned_real_receipt_schema(kind, change):
    stage = runner.stages()[0 if kind == 'step' else 2]
    n = stage['controls']
    py = {'control_completed': n, 'native_attempted': 5*n, 'native_returned': 5*n,
          'native_failed': 0, 'forbidden_entries': 0}
    result = {'completed_controls': n, 'python': py, 'warnings': [], 'input_reset_used': False,
              'terminated': False, 'truncated': True, 'stop_reason': 'time_limit',
              'full_comparison_executed': True, 'cases': [{}, {}]}
    receipt = {'schema': 'd1-true15mm-worker-35-v1' if kind == 'step' else 'd1-c23-gui-worker-receipt-v1',
               'execution_contract_id': runner.CONTRACT, 'execution_complete': True, 'failure': None,
               'cleanup_errors': [], 'warnings': [], 'archive_failed': False, 'python': py, 'result': result}
    if change == 'old_contract':receipt['execution_contract_id'] = 'C24_true_15mm_B22_v1'
    elif change == 'schema':receipt['schema'] = 'd1-true15mm-worker-24-v1'
    elif change == 'native':py['native_returned'] -= 1
    elif change == 'warning':receipt['warnings'] = ['bad']
    if change is None:assert runner.case_completion(stage, receipt) == n
    else:
        with pytest.raises(ValueError):runner.case_completion(stage, receipt)


def test_actual_isolated_pythonpath_resolves_new_entry_imports_without_executing_modules():
    config = runner.c.read_json(ROOT/runner.c.MANIFEST)
    canonical = runner.REGISTERED_ROOT
    paths = []
    for item in config['runtime_environment']['PYTHONPATH'].split(':'):
        path = Path(item)
        paths.append(str(ROOT/path.relative_to(canonical)) if path.is_relative_to(canonical) else item)
    relative = {'input23': 'continuation23/gui23/input23.py', 'native_guard24': 'continuation24/native_guard24.py',
                'env24': 'continuation24/env24.py', 'geometry24': 'continuation24/geometry24.py',
                'kinematics24': 'continuation24/kinematics24.py'}
    for name, suffix in relative.items():
        spec = importlib.machinery.PathFinder.find_spec(name, [str(ROOT/'scripts'), *paths])
        target = ROOT/'runtime/d1_b22/research'/suffix
        assert spec is not None and Path(spec.origin) == target
        canonical_name = str(canonical/target.relative_to(ROOT))
        assert config['source_hashes'][canonical_name] == runner.c.identity(target)


@pytest.mark.parametrize('fault', [None, 'natural_short', 'partial_exit', 'missing_receipt', 'cleanup', 'resource', 'late', 'signal_finalize', 'write_finalize'])
def test_actual_outer_serial_lifecycle_preserves_task_failures_and_stops_engineering_errors(tmp_path, monkeypatch, numeric_cases, fault):
    fixtures, scorer, originals, baseline = numeric_cases
    output = tmp_path/'results/b22_gain_suite35_synthetic'
    output.parent.mkdir()
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/b22_suite31_contract.py').write_bytes((ROOT/'scripts/b22_suite31_contract.py').read_bytes())
    state = SimpleNamespace(now=0., memory=8 << 30, launched=[], reaper=False, saved=[])
    reports = {item['stage']: copy.deepcopy(item) for item in originals}
    if fault == 'natural_short':
        stage = runner.stages()[2]
        item = reports[stage['name']]
        source = fixtures.canonical(stage, 'final_policy', error=.0001, n=17, failure=True)
        item['actual_counts'].update(controls=17, normal_native=85)
        item['gain_ledger_rows'] = 17
        item['cases'] = [scorer.score_case(source, expected_seed=stage['seed'], mirror=False)]
    host = SimpleNamespace(subreaper=lambda: setattr(state, 'reaper', True),
        clean_descendants=lambda *a: {'remaining': {'owned': 'live'} if fault == 'cleanup' else {}})
    config = {'host': 'synthetic-host', 'runtime_environment': {}, 'shared_session': {},
              'model_limit_keys': ['load', 'torch_load', 'predict', 'learn', 'train', 'save']}
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'REGISTERED_ROOT', tmp_path)
    monkeypatch.setattr(runner, 'source_identity', lambda: {'git_commit': 'synthetic', 'authored': {}})
    monkeypatch.setattr(runner, 'read_trigger', lambda: {'inputs': {}})
    monkeypatch.setattr(runner, 'closure', lambda *a: {'synthetic-host': {}, str(tmp_path/'scripts/read_b22_gain_suite35.py'): {'sha256': 'reader'}})
    monkeypatch.setattr(runner, 'validate_release', runner.c.read_json)
    monkeypatch.setattr(runner.c, 'registered_config', lambda: (config, {'status': 'SYNTHETIC'}))
    monkeypatch.setattr(runner.c, 'load_pinned', lambda *a: host)
    monkeypatch.setattr(runner.common, 'memory', lambda: state.memory)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda path: SimpleNamespace(free=4 << 30))
    monkeypatch.setattr(runner.os, 'getpriority', lambda *a: 10)
    monkeypatch.setattr(runner.time, 'monotonic', lambda: state.now)
    original_identity = runner.c.identity
    armed = [True]
    def identity(path):
        if fault in ('signal_finalize', 'write_finalize') and Path(path).name == 'step_pair_readback.json' and armed[0]:
            armed[0] = False
            if fault == 'signal_finalize':signal.raise_signal(signal.SIGTERM)
            raise OSError('synthetic write failure after stage fact is already counted')
        return original_identity(path)
    monkeypatch.setattr(runner.c, 'identity', identity)
    def supervise(argv, log, **kwargs):
        assert state.reaper
        state.now += 1.
        if '--request' in argv:
            request = runner.c.read_json(argv[argv.index('--request')+1])
            name = request['arm']
            stage = next(s for s in runner.stages() if s['name'] == name)
            state.launched.append(name)
            run = output/name
            run.mkdir()
            if fault == 'missing_receipt':return 1
            release = runner.c.read_json(output/'release.json')
            session = {**request['spec'], 'source_hashes': release['inputs']}
            runner.c.write_json(run/'session.json', session)
            n = 17 if fault == 'partial_exit' else reports[name]['actual_counts']['controls']
            py = {'control_completed': n, 'native_attempted': 5*n, 'native_returned': 5*n,
                  'native_failed': 0, 'forbidden_entries': 0}
            result = {'completed_controls': n, 'python': py, 'C_final': {'construction_returns': 2},
                      'warnings': [], 'input_reset_used': False, 'terminated': n != stage['controls'],
                      'truncated': n == stage['controls'], 'stop_reason': 'time_limit' if n == stage['controls'] else 'fall_or_low_clearance',
                      'full_comparison_executed': True, 'cases': [{}, {}]}
            receipt = {'schema': 'd1-true15mm-worker-35-v1' if stage['kind'] == 'step' else 'd1-c23-gui-worker-receipt-v1',
                       'execution_contract_id': runner.CONTRACT, 'session_identity': runner.c.identity(run/'session.json'),
                       'execution_complete': True, 'failure': None, 'cleanup_errors': [], 'warnings': [],
                       'archive_failed': False, 'result': result, 'python': py, 'C_final': {'construction_returns': 2}}
            runner.c.write_json(run/'worker_receipt.json', receipt)
            if fault == 'resource':state.memory = 3 << 30
            kwargs['sample'](False)
            return 1 if fault == 'partial_exit' else 0
        if '--stage' in argv:
            name = argv[argv.index('--stage')+1]
            item = copy.deepcopy(reports[name])
            item.update(release_identity=runner.c.identity(output/'release.json'), reader_identity={'sha256': 'reader'})
            runner.c.write_json(Path(argv[argv.index('--report')+1]), item)
            state.saved.append(item)
        else:
            compared = reader.aggregate(copy.deepcopy(state.saved), scorer.score_pairs, baseline)
            compared.update(schema='b22-gain-suite35-comparison-v1', release_identity=runner.c.identity(output/'release.json'),
                            stage_reports={s['name']+'_readback.json': runner.c.identity(output/(s['name']+'_readback.json')) for s in runner.stages()})
            runner.c.write_json(output/'comparison.json', compared)
            if fault == 'late':state.now = 1801.
        kwargs['sample'](False)
        return 0
    monkeypatch.setattr(runner.common, 'supervise', supervise)
    result = runner.execute(output)
    if fault in (None, 'natural_short'):
        assert result['status'] == 'COMPLETE' and len(state.launched) == 15
        assert result['actual_counts']['controls'] == (25200 if fault is None else 23617)
        assert result['RL_contribution_passed'] is (fault is None)
    else:
        assert result['status'] == 'INCOMPLETE' and result['actual_counts'] is None
        if fault not in ('cleanup', 'late'):assert state.launched == ['step_pair']
        if fault == 'partial_exit':assert result['known_actual_counts'] == {'controls': 17, 'normal_native': 85, 'compiler_native': 2}
        if fault in ('signal_finalize', 'write_finalize'):
            assert result['known_actual_counts'] == {'controls': 2400, 'normal_native': 12000, 'compiler_native': 2}
        if fault == 'missing_receipt':assert result['active_stage_actual_counts'] is None
        assert result['finite_suite_task_qualification_passed'] is None


def test_counted_prefix_is_idempotent_across_the_stage_finalization_boundary():
    stage = runner.stages()[0]
    fact = {'controls': 2400, 'normal_native': 12000, 'compiler_native': 2}
    assert runner.counted_prefix({}, stage, fact) == fact
    assert runner.counted_prefix({stage['name']: fact}, stage, fact) == fact
    assert runner.counted_prefix({stage['name']: fact}) == fact
