"""Pure gain/trigger/AST fixtures; never import a real worker or policy."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = module('test_gain34_runner', 'scripts/run_b22_gain34.py')
reader = module('test_gain34_reader', 'scripts/read_b22_gain34.py')


def score(stage, sse=10., cost=1., n=1600):
    return {'case_id': stage['case_id'], 'actor': 'final_policy', 'seed': stage['seed'], 'mirror': False,
            'completed_controls': n, 'task_passed': n == 1600, 'record': {'full_horizon': n == 1600},
            'windows': {'control_cap': 1600, 'hold': stage['hold'], 'release_tick_t0': stage['release_tick'],
                        'drive': [175, stage['release_tick']], 'final': [1500, 1600], 'final_observed_ticks': 100 if n == 1600 else 0},
            'rl_terms': {'sse_total': sse, 'torque_cost_mean': cost}}


def reports():
    return [{'status': 'COMPLETE', 'stage': stage['name'], 'execution_contract_id': runner.CONTRACT,
             'gain34': stage['gain34'], 'gain_ledger_rows': 1600,
             'actual_counts': {'controls': 1600, 'normal_native': 8000, 'compiler_native': 2},
             'pairing_identity': {'five_array_bytes': 'same', 'complete_reset': 'same'},
             'cases': [score(stage, 10. if stage['gain34'] == 1. else 8., 1. if stage['gain34'] == 1. else .9)]}
            for stage in runner.stages()]


def trigger_fixture():
    indexed, scores = {}, []
    for stage in runner.c.stages():
        item = {'status': 'COMPLETE', 'stage': stage['name']}
        if stage['kind'] == 'script':
            row = score(stage, 10. if stage['actor'] == 'zero' else 11., 1. if stage['actor'] == 'zero' else 1.1)
            row['actor'] = 'zero' if stage['actor'] == 'zero' else 'final_policy'
            row['mirror'] = stage['mirror']
            item.update(cases=[row], pairing_identity={'same_reset': True})
            scores.append(row)
        indexed[stage['name']] = item
    outer = {'status': 'COMPLETE'}
    comparison = {'status': 'COMPLETE', 'RL_contribution_passed': False, 'script_cases': scores,
                  'original_six_contribution': {'prerequisites': {'passed': True}, 'RL_contribution_passed': False}}
    return outer, comparison, indexed


def gain_row(gain):
    network = np.linspace(-.8, .8, 16, dtype=np.float32)
    action = network*np.float32(gain)
    return {'network_output16': network.tolist(), 'gain34': gain, 'action16': action.tolist(),
            'raw_action16': action.tolist(), 'policy_input_action': action.tolist(), 'policy_predict_called': True}


def test_fixed_four_new_sessions_budget_and_memory():
    stages = runner.stages()
    assert [stage['gain34'] for stage in stages] == [1., .5, 1., .5]
    assert [stage['seed'] for stage in stages] == [181202, 181202, 181203, 181203]
    assert len({stage['name'] for stage in stages}) == 4
    assert sum(stage['controls'] for stage in stages) == 6400
    assert sum(stage['controls']*5 for stage in stages) == 32000
    assert sum(stage['compiler_native'] for stage in stages) == 8
    assert runner.LIMITS['total_wall_s'] == 900
    assert runner.LIMITS['start_effective_bytes'] == 6 << 30
    assert runner.LIMITS['reserve_effective_bytes'] == 4 << 30


def test_trigger_uses_selected_two_profiles_without_requiring_other_terrain_passes():
    outer, comparison, indexed = trigger_fixture()
    indexed['rough_0p35_zero']['cases'][0]['task_passed'] = False
    indexed['rough_0p35_B']['cases'][0]['task_passed'] = False
    assert set(runner.trigger_decision(outer, comparison, indexed)) == {'flat_1p6', 'yaw_1p2'}


@pytest.mark.parametrize('change', ['partial', 'prerequisite', 'contribution', 'short', 'task', 'SSE', 'cost', 'pair', 'missing'])
def test_trigger_rejects_unestablished_or_irrelevant_conditions(change):
    outer, comparison, indexed = trigger_fixture()
    if change == 'partial':
        outer['status'] = 'INCOMPLETE'
    elif change == 'prerequisite':
        comparison['original_six_contribution']['prerequisites']['passed'] = False
    elif change == 'contribution':
        comparison['RL_contribution_passed'] = True
    elif change == 'short':
        indexed['flat_1p6_B']['cases'][0]['record']['full_horizon'] = False
    elif change == 'task':
        indexed['yaw_1p2_zero']['cases'][0]['task_passed'] = False
    elif change in ('SSE', 'cost'):
        key = 'sse_total' if change == 'SSE' else 'torque_cost_mean'
        indexed['flat_1p6_B']['cases'][0]['rl_terms'][key] = .9
    elif change == 'pair':
        indexed['flat_1p6_B']['pairing_identity'] = {}
    else:
        indexed.pop('step_pair')
    with pytest.raises(ValueError):
        runner.trigger_decision(outer, comparison, indexed)


@pytest.mark.parametrize('gain', [1., .5])
def test_gain_ledger_matches_actual_float32_controller_input(gain):
    assert reader.check_gain_row(gain_row(gain), gain) == 1


@pytest.mark.parametrize('change', ['gain', 'network', 'raw', 'policy_input', 'no_predict', 'nan', 'shape'])
def test_gain_ledger_does_not_confuse_network_output_and_actual_input(change):
    row = gain_row(.5)
    if change == 'gain':
        row['gain34'] = 1.
    elif change == 'network':
        row['network_output16'][0] += .1
    elif change == 'raw':
        row['raw_action16'] = row['network_output16']
    elif change == 'policy_input':
        row['policy_input_action'] = row['network_output16']
    elif change == 'no_predict':
        row['policy_predict_called'] = False
    elif change == 'nan':
        row['network_output16'][0] = float('nan')
    else:
        row['network_output16'].pop()
    with pytest.raises(ValueError):
        reader.check_gain_row(row, .5)


def test_metadata_adapter_is_explicit_c34_and_preserves_all_other_ast():
    prior = runner.pinned('gain34_test_source31', ROOT/'scripts/read_b22_suite31.py')
    adapted = reader.metadata_checker(prior)
    assert adapted.__name__ == 'script_sources34'
    constants = adapted.__code__.co_consts
    assert 'scripts/run_b22_worker34.py' in constants
    # The identity value is nested in the actual spec dict, not a fake saved C31 session.
    assert runner.CONTRACT in constants
    assert 'C31_B22_fixed_suite_v1' not in constants


def complete_metadata_fixture(tmp_path, gain, contract=runner.CONTRACT, returned=8000):
    """Synthetic files, actual case_completion and frozen archive/ledger functions.

    This exercises the WHOLE source adapter without model/native payloads or a
    multi-GB scan. Registration/config is a named fake boundary, not a claim of
    a real C34 execution or a substitute for canonical_script.
    """
    prior = runner.pinned('gain34_complete_metadata31', ROOT/'scripts/read_b22_suite31.py')
    stage = next(s for s in runner.stages() if s['profile'] == 'flat_1p6' and s['gain34'] == gain)
    run = tmp_path/stage['name']
    run.mkdir()
    library, binding, reference = (tmp_path/name for name in ('fake-library.so', 'fake-binding.py', 'reference.json'))
    library.write_bytes(b'not an executable library')
    binding.write_text('# synthetic binding identity only\n')
    construction = {'proof': {'passed': True, 'dso_path': str(library),
                              'dso_sha256': runner.c.identity(library)['sha256'],
                              'jump_slots': [{'passed': True}]*4},
                    'nominal_cache': {'misses': 1}, 'geometry_binding': {}, 'compiled_geometry': {}}
    runner.c.write_json(reference, construction)
    config = {'model_limit_keys': ['load', 'torch_load', 'predict', 'learn', 'train', 'save',
                                    'forward', 'backward', 'evaluate_actions', 'predict_values'],
              'shared_session': {'library': str(library), 'binding_module': str(binding),
                                 'reference_construction_path': str(reference), 'controller_variant': 'combined'}}
    actual_c = prior.c
    prior.c = SimpleNamespace(ROOT=ROOT, MANIFEST=actual_c.MANIFEST, require=actual_c.require,
        identity=actual_c.identity, model_limits=actual_c.model_limits, case_completion=actual_c.case_completion,
        read_json=lambda path: config if Path(path) == ROOT/actual_c.MANIFEST else actual_c.read_json(path))
    frozen_path = ROOT/'runtime/d1_b22/research/continuation26/root26/reader26.py'
    frozen_source = frozen_path.read_text()
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == 'bedcd547353ee876797bf7f96882c1761355894c664c531932202bbe3bb7f2f7'
    names = {'require', 'identity', 'reject_constant', 'document', 'committed', 'saved_document', 'check_ledgers'}
    nodes = [n for n in ast.parse(frozen_source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {'json': json, 'hashlib': hashlib, 'Path': Path, 'ARCHIVE': 'd1-archive-transaction-13-v1'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(frozen_path), 'exec'), namespace)  # noqa: S102 -- pinned pure archive/ledger functions
    base = SimpleNamespace(saved_document=namespace['saved_document'], check_ledgers=namespace['check_ledgers'])
    # Extract the pinned expected module-path table rather than importing those modules.
    source_fn = next(n for n in ast.parse((ROOT/'scripts/read_b22_suite31.py').read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'script_sources')
    relative = ast.literal_eval(next(n.value for n in source_fn.body if isinstance(n, ast.Assign)
                                     and any(isinstance(t, ast.Name) and t.id == 'relative' for t in n.targets)))
    origins = {name: str(ROOT/'runtime/d1_b22/research'/path) for name, path in relative.items()}
    worker_path = ROOT/'scripts/run_b22_worker34.py'
    origins.update(run_gui23=str(worker_path), engine_binding=str(binding))
    paths = {*(Path(p) for p in origins.values()), library, ROOT/actual_c.MANIFEST}
    sources = {str(path): actual_c.identity(path) for path in paths}
    limits = actual_c.model_limits('B', 1600, config['model_limit_keys'])
    spec = {'profile': stage['profile'], 'actor': 'B', 'seed': stage['seed'], 'mirror': False,
            'control_limit': 1600, 'mode': 'script', 'render': False, 'qualification': False,
            'retry_permitted': False, 'execution_contract_id': contract, 'controller_variant': 'combined',
            'arm': stage['name'], 'output_directory': str(run), 'gain34': gain, 'model_limits': limits}
    go_path = tmp_path/'go.json'
    go = {'decision': 'GO', 'worker': str(worker_path), 'repository': str(ROOT),
          'shared_session': config['shared_session'], 'arms': {stage['name']: spec}, 'inputs': sources}
    runner.c.write_json(go_path, go)
    session = {**config['shared_session'], **spec, 'source_hashes': sources,
               'argv': [str(worker_path), '--session', str(run/'session.json'), '--output', str(run)],
               'source_go_path': str(go_path), 'source_go_identity': actual_c.identity(go_path)}
    runner.c.write_json(run/'session.json', session)
    def saved(name, value):
        path = run/name
        runner.c.write_json(path, value)
        runner.c.write_json(path.with_name(path.name+'.manifest.json'),
                           {'schema': namespace['ARCHIVE'], 'payloads': [{'file': path.name, **actual_c.identity(path)}]})
    count = 1600
    api = {'load': 1, 'torch_load': 3, 'predict': 1601, 'learn': 0, 'train': 0, 'save': 0,
           'forward': 0, 'backward': 0, 'evaluate_actions': 0, 'predict_values': 0}
    counters = {key: {'attempted': value, 'returned': value} for key, value in api.items()}
    counters['predict'].update(rows_attempted=1632, rows_returned=1632)
    result = {'completed_controls': count, 'terminated': False, 'truncated': True, 'stop_reason': 'time_limit',
        'warnings': [], 'input_reset_used': False, 'gain34': gain, 'policy_predictions': count,
        'python': {'control_attempted': count, 'control_completed': count, 'native_attempted': 8000,
                   'native_returned': returned, 'native_failed': 0, 'forbidden_entries': 0,
                   'clock_advanced_substeps': 8000, 'fatal_latched': False},
        'C_final': {'construction_attempts': 2, 'construction_returns': 2, 'control_attempts': 8000,
                    'control_returns': 8000, 'phase': 0, 'target_model': 0, 'target_data': 0, 'violations': 0,
                    'native_construction_caller_verified': True, 'ccd_attempts': 0, 'ccd_returns': 0},
        'native_guard': {'native_attempted': 8000, 'native_returned': 8000, 'native_checked': 8000, 'failure': None},
        'thread_violations': [], 'model_calls': {'limits': limits, 'counts': counters, 'actor_rows': 1632, 'critic_rows': 0},
        'copy_edges': {'rejected': 0, 'live_to_measurement': 1600, 'measurement_to_writing': 3, 'reading_to_display': 0},
        'mailbox': {'published': 3, 'acquired': 0, 'released': 0, 'copy_failed': 0},
        'owner_thread_ident': 1, 'main_thread_ident': 2,
        'data_addresses': {'live': 3, 'measurement': 4, 'display': 5, 'slots': [6, 7, 8]},
        'control_step_caller_verified': True, 'segment_receipts': [{'synthetic': True}]}
    worker = {'schema': 'd1-c23-gui-worker-receipt-v1', 'execution_contract_id': contract,
              'arm': stage['name'], 'session_identity': actual_c.identity(run/'session.json'),
              'failure': None, 'warnings': [], 'archive_failed': False, 'writer_partial': [], 'result': result}
    saved('worker_receipt.json', worker)
    saved('construction_receipt.json', construction)
    saved('loaded_origins_final.json', {'module_origins': {'synthetic': str(worker_path)}, 'synthetic_module_aliases': {}})
    saved('runtime_module_origins.json', origins)
    for name in ('mapped_libraries_before_load.json', 'mapped_libraries_final.json'):
        saved(name, {'paths': [str(library)]})
    runner.c.write_json(run/'host_receipt.json', {'failure': None, 'exit_code': 0, 'source_mismatches': [],
        'arm': stage['name'], 'owned_no_orphans': True, 'reservation_closed': True,
        'reserved_controls': 1600, 'execution_started_monotonic': 123.})
    runner.c.write_json(run/'supervisor_receipt.json', {'failure': None, 'exit_code': 0, 'cleanup': {'remaining': {}}})
    runner.c.write_json(run/'preflight_ready.json', {'session_sha256': actual_c.identity(run/'session.json')['sha256'],
        'full_source_sha256_checked': True, 'monotonic_s': 123.})
    return reader.metadata_checker(prior), (run, stage, {'inputs': sources}, base)


@pytest.mark.parametrize('gain', [1., .5])
def test_complete_c34_metadata_path_accepts_real_case_and_archive_ledger_functions(tmp_path, gain):
    checker, args = complete_metadata_fixture(tmp_path, gain)
    session, worker, construction, count = checker(*args)
    assert count == 1600 and session['execution_contract_id'] == worker['execution_contract_id'] == runner.CONTRACT
    assert worker['result']['gain34'] == gain and construction['proof']['passed'] is True


@pytest.mark.parametrize('change', ['C31', 'count'])
def test_complete_c34_metadata_path_does_not_relabel_old_contract_or_bad_count(tmp_path, change):
    checker, args = complete_metadata_fixture(tmp_path, .5,
        contract='C31_B22_fixed_suite_v1' if change == 'C31' else runner.CONTRACT,
        returned=7999 if change == 'count' else 8000)
    with pytest.raises(ValueError, match='session identity|accounting'):
        checker(*args)


def test_worker_is_exact_explicit_31_derivative_except_declared_seams():
    source = (ROOT/'scripts/run_b22_worker31.py').read_text()
    old_doc = ast.parse(source).body[0]
    actual = ast.parse((ROOT/'scripts/run_b22_worker34.py').read_text())
    replacements = [
        ("or session.get('execution_contract_id') != 'C31_B22_fixed_suite_v1'",
         "or session.get('execution_contract_id') != 'C34_B22_residual_gain_probe_v1'"),
        ("or session.get('actor') not in ('B','zero')\n            or session.get('profile') not in PROFILE_CASE",
         "or session.get('actor') != 'B'\n            or session.get('profile') not in ('flat_1p6','yaw_1p2')\n            or type(session.get('gain34')) is not float or session['gain34'] not in (1.0,.5)"),
        ("not 1 <= session['control_limit'] <= 6000", "session['control_limit'] != 1600"),
        ("action,predicted=_numeric_actor_action(\n                                'zero' if policy is None else 'final_policy',policy,observation)",
         "network_output16,predicted=_numeric_actor_action(\n                                'zero' if policy is None else 'final_policy',policy,observation)\n                            network_output16=np.asarray(network_output16,dtype=np.float32).copy()\n                            action=network_output16*np.float32(session['gain34'])"),
        ("'input_observation99':input_obs,'raw_action16':action,",
         "'input_observation99':input_obs,'raw_action16':action,\n                                'network_output16':network_output16,'gain34':session['gain34'],"),
        ("shared['result']={'completed_controls':completed,",
         "shared['result']={'completed_controls':completed,'gain34':session['gain34'],"),
    ]
    for old, new in replacements:
        assert source.count(old) == 1
        source = source.replace(old, new)
    expected = ast.parse(source)
    assert isinstance(old_doc, ast.Expr)
    expected.body[0] = actual.body[0]
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


@pytest.mark.parametrize('gain', [1., .5])
@pytest.mark.parametrize('sample', ['normal', 'edges'])
def test_actual_worker_action_seam_with_fake_predictor_is_exact_and_does_not_mutate_output(gain, sample):
    tree = ast.parse((ROOT/'scripts/run_b22_worker34.py').read_text())
    owner = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'owner')
    assignments = [node for node in ast.walk(owner) if isinstance(node, ast.Assign)]
    selected = [node for node in assignments if isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == '_numeric_actor_action']
    assert len(selected) == 1
    first = selected[0]
    following = sorted((node for node in assignments if first.lineno <= node.lineno <= first.end_lineno+2), key=lambda n: n.lineno)
    assert len(following) == 3
    original = np.linspace(-.9, .9, 16, dtype=np.float32)
    if sample == 'edges':
        tiny = np.nextafter(np.float32(0.), np.float32(1.))
        original = np.asarray([-1., 1., -0., 0., tiny, -tiny, 2*tiny, -2*tiny]*2, dtype=np.float32)
    before = original.copy()
    namespace = {'_numeric_actor_action': lambda *args: (original, True), 'np': np,
                 'policy': object(), 'observation': np.zeros(99, np.float32), 'session': {'gain34': gain}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=following, type_ignores=[])), '<actual-worker34-action-seam>', 'exec'), namespace)  # noqa: S102 -- fixed local worker AST, fake predictor
    assert original.tobytes() == before.tobytes()
    assert namespace['network_output16'].dtype == np.float32
    assert namespace['action'].dtype == np.float32
    assert namespace['action'].tobytes() == (before*np.float32(gain)).tobytes()


@pytest.mark.parametrize('gain,valid', [(1., True), (.5, True), (1, False), (True, False), (2., False), (0., False)])
def test_actual_worker_admission_allows_only_the_two_float_gains(tmp_path, gain, valid):
    tree = ast.parse((ROOT/'scripts/run_b22_worker34.py').read_text())
    preflight = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'preflight')
    session = {'schema': 'd1-c23-gui-worker-v1', 'execution_contract_id': runner.CONTRACT,
               'mode': 'script', 'render': False, 'qualification': False, 'actor': 'B',
               'profile': 'flat_1p6', 'gain34': gain, 'control_limit': 1600,
               'retry_permitted': False, 'output_directory': str(tmp_path), 'argv': ['fake']}
    namespace = {'session': session, 'session_path': tmp_path/'session.json', 'output': tmp_path,
                 'sys': SimpleNamespace(argv=['fake'])}
    body = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(preflight.body[0])], type_ignores=[]))
    if valid:
        exec(compile(body, '<actual-worker34-admission>', 'exec'), namespace)  # noqa: S102 -- fixed local guard AST
    else:
        with pytest.raises(RuntimeError, match='session/argv'):
            exec(compile(body, '<actual-worker34-admission>', 'exec'), namespace)  # noqa: S102 -- fixed local guard AST


@pytest.mark.parametrize('available,admission,rejected', [
    ((4 << 30)-1, False, True), (4 << 30, False, False),
    ((6 << 30)-1, True, True), (6 << 30, True, False),
])
def test_actual_sampler_keeps_four_gib_and_six_gib_admission(tmp_path, available, admission, rejected):
    tree = ast.parse((ROOT/'scripts/run_b22_gain34.py').read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    sample = next(node for node in ast.walk(owner) if isinstance(node, ast.FunctionDef) and node.name == 'sample')
    namespace = {'common': SimpleNamespace(memory=lambda: available),
                 'shutil': SimpleNamespace(disk_usage=lambda path: SimpleNamespace(free=2 << 30)),
                 'report': {}, 'time': SimpleNamespace(monotonic=lambda: 0.), 'started': 0.,
                 'deadline': 10., 'output': tmp_path, 'json': json, 'c': runner.c}
    body = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(sample)], type_ignores=[]))
    exec(compile(body, '<actual-gain34-resource-sampler>', 'exec'), namespace)  # noqa: S102 -- fixed local sampler AST
    if rejected:
        with pytest.raises(ValueError, match='RAM'):
            namespace['sample'](admission)
    else:
        namespace['sample'](admission)


def test_new_pairs_can_report_local_improvement_without_claiming_global_adoption():
    result = reader.aggregate(reports())
    assert result['all_two_profile_SSE_and_cost_lower'] is True
    assert result['RL_contribution_claimed'] is False and result['adoption_decision'] == 'NOT_AUTOMATIC'


def test_partial_lower_error_cannot_be_scored_as_an_improvement():
    items = reports()
    items[1]['cases'] = [score(runner.stages()[1], .01, .01, n=100)]
    items[1]['actual_counts'].update(controls=100, normal_native=500)
    items[1]['gain_ledger_rows'] = 100
    result = reader.aggregate(items)
    assert result['all_two_profile_SSE_and_cost_lower'] is False
    assert all(item['SSE_gain0p5_over_gain1'] is None and item['cost_gain0p5_over_gain1'] is None for item in result['pairs'])


def test_short_episode_cannot_be_relabelled_as_full_horizon():
    items = reports()
    items[1]['cases'] = [score(runner.stages()[1], .01, .01, n=100)]
    items[1]['cases'][0]['record']['full_horizon'] = True
    items[1]['actual_counts'].update(controls=100, normal_native=500)
    items[1]['gain_ledger_rows'] = 100
    with pytest.raises(ValueError, match='full-horizon'):
        reader.aggregate(items)


def test_zero_denominator_does_not_claim_defined_relative_improvement():
    items = reports()
    items[0]['cases'][0]['rl_terms']['sse_total'] = 0.
    result = reader.aggregate(items)
    assert result['full_window_ratios_available'] is False
    assert result['all_two_profile_SSE_and_cost_lower'] is False


@pytest.mark.parametrize('change', ['pair', 'gain', 'actor', 'profile', 'missing', 'order', 'native'])
def test_new_pair_identity_and_matrix_are_required(change):
    items = reports()
    if change == 'pair':
        items[1]['pairing_identity'] = {}
    elif change == 'gain':
        items[1]['gain34'] = 1.
    elif change in ('actor', 'profile'):
        items[1]['cases'][0]['actor' if change == 'actor' else 'case_id'] = 'wrong'
    elif change == 'missing':
        items.pop()
    elif change == 'order':
        items.reverse()
    else:
        items[1]['actual_counts']['normal_native'] -= 1
    with pytest.raises(ValueError):
        reader.aggregate(items)


def test_default_plan_import_stays_cold_and_creates_no_output(tmp_path):
    path = tmp_path/'not-created'
    code = ('import importlib.util,sys;s=importlib.util.spec_from_file_location("cold34","scripts/run_b22_gain34.py");'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'assert not any(x in sys.modules for x in ("numpy","torch","mujoco","glfw","stable_baselines3"));'
            f'm.main(["--output",{str(path)!r}])')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['execution_started'] is False and not path.exists()


def test_requests_bind_new_worker_gain_and_fresh_output_without_launch(tmp_path):
    config = runner.c.read_json(ROOT/runner.c.MANIFEST)
    release_path = tmp_path/'release.json'
    runner.c.write_json(release_path, {'synthetic': True})
    for stage in runner.stages():
        argv = runner.request(stage, tmp_path, {'inputs': {}}, config, release_path)
        request = runner.c.read_json(tmp_path/(stage['name']+'_launch/request.json'))
        go = runner.c.read_json(request['source_go_path'])
        assert request['spec']['execution_contract_id'] == runner.CONTRACT
        assert request['spec']['gain34'] == stage['gain34']
        assert request['spec']['output_directory'] == str(tmp_path/stage['name'])
        assert go['worker'] == str(ROOT/'scripts/run_b22_worker34.py')
        assert argv[4] == config['host'] and not (tmp_path/stage['name']).exists()


@pytest.mark.parametrize('change', [None, 'old_contract', 'gain', 'reader', 'release', 'missing', 'zero', 'too_many', 'native', 'compiler', 'ledger'])
def test_outer_stage_gate_rejects_count_and_identity_errors_before_next_session(change):
    stage = runner.stages()[0]
    rid, source = {'sha256': 'release'}, {'sha256': 'reader'}
    item = {'schema': 'b22-gain34-stage-readback-v1', 'status': 'COMPLETE', 'stage': stage['name'],
            'execution_contract_id': runner.CONTRACT, 'gain34': 1., 'release_identity': rid,
            'reader_identity': source, 'actual_counts': {'controls': 1600, 'normal_native': 8000, 'compiler_native': 2},
            'gain_ledger_rows': 1600}
    if change == 'old_contract':
        item['execution_contract_id'] = 'C31_B22_fixed_suite_v1'
    elif change == 'gain':
        item['gain34'] = .5
    elif change in ('reader', 'release'):
        item[change+'_identity'] = {}
    elif change == 'missing':
        item.pop('actual_counts')
    elif change in ('zero', 'too_many'):
        item['actual_counts']['controls'] = 0 if change == 'zero' else 1601
    elif change in ('native', 'compiler'):
        item['actual_counts']['normal_native' if change == 'native' else 'compiler_native'] -= 1
    elif change == 'ledger':
        item['gain_ledger_rows'] -= 1
    if change is None:
        assert runner.stage_counts(stage, item, rid, source) == item['actual_counts']
    else:
        with pytest.raises(ValueError, match='identity/actual counters'):
            runner.stage_counts(stage, item, rid, source)


def test_subreaper_precedes_all_worker_and_saved_reader_launches():
    tree = ast.parse((ROOT/'scripts/run_b22_gain34.py').read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    calls = [n for n in ast.walk(owner) if isinstance(n, ast.Call)]
    reaper = next(n for n in calls if ast.unparse(n.func) == 'host.subreaper')
    launches = [n for n in calls if ast.unparse(n.func) == 'common.supervise']
    assert len(launches) == 3 and all(reaper.lineno < n.lineno for n in launches)
    assert not any(ast.unparse(n.func) == 'subprocess.Popen' for n in calls)
