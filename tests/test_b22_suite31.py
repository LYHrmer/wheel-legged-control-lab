"""Small synthetic protocol/stream/scoring tests; no model or physical step."""
from __future__ import annotations

import ast
import gzip
import hashlib
import importlib.machinery
import importlib.util
import json
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


contract = module('test_contract31', 'scripts/b22_suite31_contract.py')
runner = module('test_runner31', 'scripts/run_b22_suite31.py')
reader = module('test_reader31', 'scripts/read_b22_suite31.py')


@pytest.fixture(scope='module')
def score18():
    before = list(sys.path)
    try:
        sys.path[:0] = [str(ROOT/'runtime/d1_b22/research/course_impl08')]
        yield module('test_pure_score31', 'runtime/d1_b22/research/continuation18/score18.py')
    finally:
        sys.path[:] = before


def canonical(stage, actor, error=.01, n=None, failure=False):
    cap = stage['controls']
    n = cap if n is None else n
    vx = yaw = 0.
    servo_vx, servo_yaw = [], []
    for tick in range(n):
        raw = contract.raw_command(stage, tick)
        vx += max(-.005, min(.005, raw['forward_velocity_mps']-vx))
        yaw += max(-.006, min(.006, raw['yaw_rate_rps']-yaw))
        servo_vx.append(vx)
        servo_yaw.append(yaw)
    servo_vx, servo_yaw = np.asarray(servo_vx), np.asarray(servo_yaw)
    actual = np.where(np.arange(n) < stage['release_tick'], servo_vx+error, 0.)
    actual[:175] = 0.
    projection = np.cumsum(actual)*.01
    d = {'case_id': stage['case_id'], 'actor': actor, 'seed': stage['seed'],
         'mirror': stage['mirror'], 'terrain': stage['terrain'],
         'task_schema': 'd1-course-world-upright-rl16-task-v1',
         'reference_schema': 'd1-course-world-upright-roll-pitch-zero-v1',
         'reward_schema': 'd1-course-world-upright-bodycom-yaw-terrain-action-torque-v1',
         'completed_controls': n, 'terminated': failure, 'truncated': not failure,
         'stop_reason': 'fall_or_low_clearance' if failure else 'time_limit',
         'tick_index': np.arange(n), 'native_index': np.arange(5*n),
         'applied_servo_vx_mps': servo_vx, 'applied_servo_yaw_rps': servo_yaw,
         'com_vx_mps': actual, 'body_yaw_rate_rps': servo_yaw,
         'heading_rad': np.cumsum(servo_yaw)*.01, 'clearance_m': np.full(n, .455),
         'lateral_offset_m': np.zeros(n), 'forward_projection_m': projection,
         'motor_torque_nm': np.ones((n, 16, 5)),
         'initial_base_x_m': 0., 'initial_base_y_m': 0., 'initial_clearance_m': .455,
         'initial_roll_deg': 0., 'initial_pitch_deg': 0., 'initial_terrain_relative_tilt_deg': 0.,
         'native_clearance_m': np.full(5*n, .455),
         'native_forward_projection_m': np.repeat(projection, 5),
         'native_nonwheel_ground_candidate_count': np.zeros(5*n, dtype=int),
         'positive_wheel_load_native_counts_by_family': {'floor': 5*n, 'bumps': 1, 'rough': 1, 'ramp': 1},
         'warning_count': 0, 'geometry_invalid_count': 0, 'map_escape_count': 0, 'fall_count': int(failure),
         'ramp_positive_wheel_load_native_counts_by_geom': {k: 1 for k in ('terrain_ramp_up', 'terrain_ramp_deck', 'terrain_ramp_down')},
         'ramp_geom_world_x_max_m_by_name': {k: 0. for k in ('terrain_ramp_up', 'terrain_ramp_deck', 'terrain_ramp_down')},
         'wheel_collision_world_x_min_m_by_index': {str(k): 1. for k in range(4)}}
    for key in ('native_roll_deg', 'native_pitch_deg', 'native_terrain_relative_tilt_deg', 'native_base_x_m', 'native_base_y_m'):
        d[key] = np.zeros(5*n)
    return d


def reports(score18, policy_error=.005):
    items = []
    for stage in contract.stages():
        report = {'status': 'COMPLETE', 'stage': stage['name'],
                  'actual_counts': {'controls': stage['controls'], 'normal_native': 5*stage['controls'], 'compiler_native': 2},
                  'cases': [{'actor': 'zero', 'task_qualified': True}, {'actor': 'B', 'task_qualified': True}]}
        if stage['kind'] == 'script':
            source = canonical(stage, 'zero' if stage['actor'] == 'zero' else 'final_policy',
                               .01 if stage['actor'] == 'zero' else policy_error)
            report['cases'] = [score18.score_case(source, expected_seed=stage['seed'], mirror=stage['mirror'])]
            report['pairing_identity'] = {'five_arrays': 'synthetic-identical', 'complete_control_reset': 'synthetic-identical'}
        items.append(report)
    return items


def worker(stage, n=None, failure=False):
    n = stage['controls'] if n is None else n
    return {'schema': 'd1-c23-gui-worker-receipt-v1', 'failure': None, 'warnings': [], 'archive_failed': False,
            'result': {'completed_controls': n, 'warnings': [], 'input_reset_used': False,
                       'terminated': failure, 'truncated': not failure,
                       'stop_reason': 'fall_or_low_clearance' if failure else 'time_limit',
                       'python': {'control_completed': n, 'native_attempted': 5*n, 'native_returned': 5*n,
                                  'native_failed': 0, 'forbidden_entries': 0}}}


def test_fixed_budget_includes_all_pairs_and_shared_step_compilation():
    stages = contract.stages()
    assert len(stages) == 15
    assert sum(s['controls'] for s in stages) == 25200
    assert sum(s['compiler_native'] for s in stages) == 30
    assert sum(s['controls']*5 for s in stages) == 126000
    assert contract.plan()['limits']['logical_arm_records'] == 16
    assert len({s['name'] for s in stages}) == len(stages)
    assert stages[0]['name'] == 'step_pair'


@pytest.mark.parametrize('index', range(7))
def test_fixed_command_horizon_hold_release_and_mirror(index):
    stage = contract.stages()[1+2*index]
    rows = [contract.raw_command(stage, i) for i in range(stage['controls'])]
    assert all(row['forward_velocity_mps'] == 0 for row in rows[:175])
    assert all(row['forward_velocity_mps'] == stage['speed_mps'] for row in rows[175:stage['release_tick']])
    assert all(row['forward_velocity_mps'] == 0 for row in rows[stage['release_tick']:])
    if stage['yaw_amplitude_rps']:
        begin = stage['hold'][0]
        sign = -1 if stage['mirror'] else 1
        assert [rows[begin+i]['yaw_rate_rps'] for i in (0, 100, 300)] == [sign*.3, -sign*.3, sign*.3]


def test_forward_rows_streams_slices_and_retains_only_last_record():
    value = reader.ForwardRows(lambda: ({'index': i} for i in range(20)), 20)
    for start in range(0, 20, 5):
        assert [x['index'] for x in value[start:start+5]] == list(range(start, start+5))
    assert value[-1] == {'index': 19}
    assert not any(isinstance(v, list) for v in vars(value).values())


@pytest.mark.parametrize('real,declared', [(4, 5), (6, 5)])
def test_stream_rejects_missing_or_extra_native_rows(real, declared):
    value = reader.ForwardRows(lambda: iter(range(real)), declared)
    with pytest.raises(ValueError, match='fewer|extra'):
        value[0:declared]


def test_stream_rejects_reordered_slice_and_second_iteration():
    value = reader.ForwardRows(lambda: iter(range(5)), 5)
    with pytest.raises(ValueError, match='ascending'):
        value[1:5]
    assert list(value) == list(range(5))
    with pytest.raises(ValueError, match='twice'):
        list(value)


def test_json_blocks_check_actual_counts_and_objects(tmp_path):
    path = tmp_path/'rows.gz'
    with gzip.open(path, 'wt') as stream:
        stream.write('{"a":1}\n{"a":2}\n')
    assert len(list(reader.json_rows([path], [2]))) == 2
    with pytest.raises(ValueError, match='row count'):
        list(reader.json_rows([path], [3]))
    with gzip.open(path, 'wt') as stream:
        stream.write('[1]\n')
    with pytest.raises(ValueError, match='nonobject'):
        list(reader.json_rows([path]))


def test_legitimate_short_task_is_collectable_but_unknown_stop_is_not():
    stage = contract.stages()[1]
    assert contract.case_completion(stage, worker(stage, 12, True)) == 12
    with pytest.raises(ValueError, match='ending'):
        contract.case_completion(stage, worker(stage, 12, False))
    broken = worker(stage, 12, True)
    broken['result']['stop_reason'] = 'owner_soft_deadline'
    with pytest.raises(ValueError, match='ending'):
        contract.case_completion(stage, broken)


@pytest.mark.parametrize('key,value', [('failure', {'message': 'bad'}), ('archive_failed', True), ('warnings', ['solver'])])
def test_engineering_failure_is_not_a_valid_task(key, value):
    stage = contract.stages()[1]
    bad = worker(stage)
    bad[key] = value
    with pytest.raises(ValueError, match='engineering'):
        contract.case_completion(stage, bad)


def test_wrong_actual_native_count_is_rejected():
    stage = contract.stages()[1]
    bad = worker(stage)
    bad['result']['python']['native_returned'] -= 1
    with pytest.raises(ValueError, match='accounting'):
        contract.case_completion(stage, bad)


def test_task_failure_still_collects_all_planned_stages(score18):
    items = reports(score18)
    items[1]['cases'][0]['task_passed'] = False
    visited = []
    def collect(stage):
        visited.append(stage['name'])
        return items[len(visited)-1]
    assert len(runner.collect_matrix(contract.stages(), collect)) == 15
    assert len(visited) == 15


def test_engineering_failure_stops_before_next_arm(score18):
    items = reports(score18)
    items[1]['status'] = 'INCOMPLETE'
    visited = []
    def collect(stage):
        visited.append(stage['name'])
        return items[len(visited)-1]
    with pytest.raises(ValueError, match='incomplete'):
        runner.collect_matrix(contract.stages(), collect)
    assert visited == ['step_pair', 'flat_0p6_zero']


def test_real_frozen_contribution_formula_can_pass_and_fail(score18):
    good = reader.aggregate(reports(score18), score18.score_pairs)
    assert good['RL_contribution_passed'] is True
    assert good['original_six_contribution']['branch_b_sse_improvement']['pooled_sse_ratio'] == pytest.approx(.25)
    bad = reader.aggregate(reports(score18, .02), score18.score_pairs)
    assert bad['RL_contribution_passed'] is False
    assert len(good['mirror_scores_separate']) == 2
    assert len(good['real_15mm_scores_separate']) == 2


def test_incomplete_policy_cannot_gain_from_low_short_error(score18):
    items = reports(score18)
    stage = contract.stages()[2]
    source = canonical(stage, 'final_policy', 0., n=100, failure=True)
    items[2]['cases'] = [score18.score_case(source, expected_seed=stage['seed'])]
    items[2]['actual_counts'].update(controls=100, normal_native=500)
    result = reader.aggregate(items, score18.score_pairs)
    assert result['RL_contribution_passed'] is False
    assert not result['all_seven_B_script_tasks_passed']


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'reordered', 'reset'])
def test_matrix_and_exact_reset_pair_identity_are_required(score18, change):
    items = reports(score18)
    if change == 'missing':
        items.pop()
    elif change == 'duplicate':
        items[-1] = items[-2]
    elif change == 'reordered':
        items[1], items[2] = items[2], items[1]
    else:
        items[2]['pairing_identity']['complete_control_reset'] = 'different'
    with pytest.raises(ValueError, match='matrix|pairing'):
        reader.aggregate(items, score18.score_pairs)


def test_comparison_cannot_replace_b_actor_or_mirror(score18):
    items = reports(score18)
    items[2]['cases'][0]['actor'] = 'zero'
    with pytest.raises(ValueError, match='actor'):
        reader.aggregate(items, score18.score_pairs)


def test_numeric_pairing_hash_checks_bytes_dtype_and_complete_memory():
    original = {'x': np.array([1., 2.], dtype=np.float32)}
    first = reader.numeric_identity(original, {'pi': [0.], 'z': 0.})
    assert first != reader.numeric_identity({'x': original['x'].astype(np.float64)}, {'pi': [0.], 'z': 0.})
    assert first != reader.numeric_identity(original, {'pi': [0.], 'z': .01})


def test_plan_and_import_are_cold_and_do_not_create_output(tmp_path):
    destination = tmp_path/'must_not_exist'
    code = ('import importlib.util,sys; p="scripts/run_b22_suite31.py"; '
            's=importlib.util.spec_from_file_location("cold31",p);m=importlib.util.module_from_spec(s);'
            's.loader.exec_module(m); assert not any(x in sys.modules for x in '
            '("torch","mujoco","numpy","onnxruntime","stable_baselines3")); '
            f'm.main(["--output",{str(destination)!r}])')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['limits']['controls'] == 25200
    assert not destination.exists()


def test_reader_import_is_cold_without_loading_legacy_reader():
    code = ('import importlib.util,sys;'
            's=importlib.util.spec_from_file_location("cold_reader31","scripts/read_b22_suite31.py");'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'assert not any(x in sys.modules for x in '
            '("torch","mujoco","onnxruntime","numpy","stable_baselines3","gymnasium","suite31_frozen_reader26"))')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def prepared_release(tmp_path):
    manifest = contract.read_json(ROOT/contract.MANIFEST)
    authored = {str(ROOT/name): contract.identity(ROOT/name) for name in contract.AUTHORED}
    value = {'schema': 'b22-suite31-release-v1', 'repository': str(ROOT),
             'plan': contract.plan(ROOT/'results/b22_suite31_synthetic_no_run'),
             'source': {'git_commit': 'a'*40, 'authored': authored},
             'inputs': {**manifest['source_hashes'], **authored, str(ROOT/contract.MANIFEST): contract.identity(ROOT/contract.MANIFEST)},
             'v27_manifest': contract.identity(ROOT/contract.MANIFEST)}
    return value, tmp_path/'release.json'


@pytest.mark.parametrize('change', ['omit_old_source', 'unknown_source', 'omit_new_source', 'horizon', 'prior_output', 'manifest'])
def test_release_rejects_omitted_or_unexplained_scope(tmp_path, change):
    value, path = prepared_release(tmp_path)
    if change == 'omit_old_source':
        value['inputs'].pop(next(iter(value['inputs'])))
    elif change == 'unknown_source':
        value['inputs']['/tmp/unexplained'] = {'bytes': 0, 'sha256': '0'*64}
    elif change == 'omit_new_source':
        value['source']['authored'].pop(next(iter(value['source']['authored'])))
    elif change == 'horizon':
        value['plan']['stages'][1]['controls'] -= 1
    elif change == 'prior_output':
        value['plan']['output'] = str(ROOT/'results/my_course_drive_01')
    else:
        value['v27_manifest']['sha256'] = '0'*64
    contract.write_json(path, value)
    with pytest.raises(ValueError):
        contract.validate_release(path, check_sources=False)


def test_scope_validation_is_read_only_without_full_input_scan(tmp_path):
    value, path = prepared_release(tmp_path)
    contract.write_json(path, value)
    assert contract.validate_release(path, check_sources=False) == value
    assert not Path(value['plan']['output']).exists()


def fake_process(polls=1, returncode=0):
    process = SimpleNamespace(returncode=returncode, remaining=polls)
    def poll():
        if process.remaining:
            process.remaining -= 1
            return None
        return returncode
    process.poll = poll
    return process


@pytest.mark.parametrize('failure', ['RAM', 'wall', 'spawn', 'signal_before_handle'])
def test_supervision_failure_cleans_only_its_owned_family(tmp_path, failure):
    cleaned, spawned = [], []
    def cleanup(pid, process):
        cleaned.append((pid, process))
        return {'remaining': {}}
    host = SimpleNamespace(clean_descendants=cleanup)
    process = fake_process()
    def factory(*args, **kwargs):
        spawned.append(args)
        if failure == 'spawn':
            raise OSError('spawn error')
        if failure == 'signal_before_handle':
            raise InterruptedError('signal before Popen returned its handle')
        assert kwargs['start_new_session'] is True
        return process
    def sample(admission):
        if failure == 'RAM':
            raise ValueError('RAM below reserve')
    with pytest.raises((ValueError, OSError), match='RAM|wall|spawn|signal'):
        runner.supervise(['fake-only'], tmp_path/'stdout', host=host, sample=sample,
                         limit=0 if failure == 'wall' else 10, deadline=20,
                         factory=factory, clock=lambda: 1., sleeper=lambda seconds: None)
    assert len(spawned) == len(cleaned) == 1
    assert cleaned[0][1] is (None if failure in ('spawn', 'signal_before_handle') else process)


def test_process_exit_is_returned_and_orphan_cleanup_failure_rejects(tmp_path):
    process = fake_process(polls=0, returncode=7)
    host = SimpleNamespace(clean_descendants=lambda pid, child: {'remaining': {}})
    assert runner.supervise(['fake'], tmp_path/'one', host=host, sample=lambda admission: None,
                            limit=10, deadline=20, factory=lambda *a, **kw: process,
                            clock=lambda: 1.) == 7
    host.clean_descendants = lambda pid, child: {'remaining': {123: 456}}
    with pytest.raises(ValueError, match='cleanup'):
        runner.supervise(['fake'], tmp_path/'two', host=host, sample=lambda admission: None,
                         limit=10, deadline=20, factory=lambda *a, **kw: process, clock=lambda: 1.)


def test_completed_process_observed_after_stage_deadline_is_rejected(tmp_path):
    process = fake_process(polls=0, returncode=0)
    clocks = iter([0., 11.])
    host = SimpleNamespace(clean_descendants=lambda pid, child: {'remaining': {}})
    with pytest.raises(ValueError, match='wall'):
        runner.supervise(['fake'], tmp_path/'late', host=host, sample=lambda admission: None,
                         limit=10, deadline=20, factory=lambda *a, **kw: process,
                         clock=lambda: next(clocks))


@pytest.mark.parametrize('resource', ['RAM', 'disk'])
def test_low_admission_never_creates_output_or_spawns(monkeypatch, resource):
    output = ROOT/'results/b22_suite31_pytest_never_create'
    assert not output.exists()
    monkeypatch.setattr(runner.c, 'source_identity', lambda root: {})
    monkeypatch.setattr(runner, 'memory', lambda: (5 if resource == 'RAM' else 7) << 30)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda root: SimpleNamespace(free=1 << 30))
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **kw: pytest.fail('spawned before resource admission'))
    with pytest.raises(ValueError, match='RAM|disk'):
        runner.execute(output)
    assert not output.exists()


def test_memory_uses_smallest_visible_ancestor_remaining(tmp_path):
    proc, cg = tmp_path/'proc', tmp_path/'cgroup'
    (proc/'self').mkdir(parents=True)
    child = cg/'scope'
    child.mkdir(parents=True)
    (proc/'meminfo').write_text('MemAvailable: 8388608 kB\n')
    (proc/'self/cgroup').write_text('0::/scope\n')
    (cg/'memory.max').write_text(str(4 << 30))
    (cg/'memory.current').write_text(str(2 << 30))
    (child/'memory.max').write_text(str(2 << 30))
    (child/'memory.current').write_text(str(1 << 30))
    assert runner.memory(proc, cg) == 1 << 30
    (proc/'self/cgroup').write_text('0::/../outside\n')
    with pytest.raises(ValueError, match='scope'):
        runner.memory(proc, cg)


def test_worker31_is_exact_explicit_c26_derivative_except_reviewed_seams():
    path = ROOT/'runtime/d1_b22/research/continuation26/gui26/run_gui26.py'
    source = path.read_text()
    expected_sha = 'a138805d43cacde6405837149fb1652e04dee08f6ab47c812fa5b3a9b2017e18'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
    old = """            or session.get('execution_contract_id') not in
               ('C26_B22_GUI_owner_close_v1', 'C26_B22_user_session_v1')
            or session.get('mode') not in ('script','keyboard')"""
    new = """            or session.get('execution_contract_id') != 'C31_B22_fixed_suite_v1'
            or session.get('mode') != 'script' or session.get('render') is not False
            or session.get('qualification') is not False"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = """                                if session['mode']=='script' and completed<session['control_limit']:
                                    raise RuntimeError('fixed script terminated before reserved horizon')"""
    new = """                                if session['mode']=='script' and completed<session['control_limit']:
                                    if not (terminated and not truncated and stop_reason in (
                                            'fall_or_low_clearance', 'nonwheel_ground_contact',
                                            'course_map_boundary')):
                                        raise RuntimeError('script ended early without a legitimate task terminal')"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    expected = ast.parse(source)
    actual_text = (ROOT/'scripts/run_b22_worker31.py').read_text()
    assert expected_sha in actual_text
    actual = ast.parse(actual_text)
    # Version prose/comments have no executable effect. Every other top-level
    # statement, function and nested function is the reviewed literal derivative.
    expected.body[0] = actual.body[0]
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


def test_real_isolated_pythonpath_resolves_relocated_worker_top_level_dependency():
    config = contract.read_json(ROOT/contract.MANIFEST)
    paths = [str(ROOT/'scripts'), *config['runtime_environment']['PYTHONPATH'].split(':')]
    expected = ROOT/'runtime/d1_b22/research/continuation23/gui23/input23.py'
    # PathFinder locates the actual file without executing input23 or a worker.
    found = importlib.machinery.PathFinder.find_spec('input23', paths)
    assert found is not None and Path(found.origin) == expected
    assert str(expected.parent) in config['runtime_environment']['PYTHONPATH'].split(':')
    assert contract.identity(expected) == config['source_hashes'][str(expected)]
    assert 'PYTHONPATH' not in config['shared_session']
    host_tree = ast.parse(Path(config['host']).read_text())
    host = next(node for node in host_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    environment_loop = next(node for node in ast.walk(host) if isinstance(node, ast.For)
                            and ast.unparse(node.iter) == 'runtime_env.items()')
    assert any(isinstance(node, ast.Assign) and any(ast.unparse(target) == 'env[key]' for target in node.targets)
               for node in ast.walk(environment_loop))
    spawn = next(node for node in ast.walk(host) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == 'subprocess.Popen')
    assert environment_loop.end_lineno < spawn.lineno
    assert any(item.arg == 'env' and ast.unparse(item.value) == 'env' for item in spawn.keywords)


def test_outer_subreaper_is_enabled_before_every_supervised_worker_or_reader_launch():
    tree = ast.parse((ROOT/'scripts/run_b22_suite31.py').read_text())
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)]
    reapers = [node for node in calls if ast.unparse(node.func) == 'host.subreaper']
    launches = [node for node in calls if ast.unparse(node.func) in ('supervise', 'collect_matrix')]
    assert len(reapers) == 1 and len(launches) == 4
    assert all(reapers[0].lineno < launch.lineno for launch in launches)
    # No direct spawn in execute can bypass the supervised cleanup seam.
    assert not any(ast.unparse(node.func) == 'subprocess.Popen' for node in calls)


@pytest.mark.parametrize('actor', ['zero', 'B'])
def test_real_script_request_binds_new_worker_and_version_without_running(tmp_path, actor):
    stage = next(s for s in contract.stages() if s['kind'] == 'script' and s['actor'] == actor)
    release_path = tmp_path/'release.json'
    contract.write_json(release_path, {'synthetic': True})
    config = {'model_limit_keys': ['load', 'torch_load', 'predict', 'learn', 'train', 'save',
                                   'forward', 'backward', 'evaluate_actions', 'predict_values'],
              'runtime_environment': {}, 'shared_session': {}, 'host': 'frozen-host-not-started'}
    destination = tmp_path/stage['name']
    argv = runner.script_request(stage, {'inputs': {}}, destination, config, release_path)
    request = contract.read_json(destination.with_name(destination.name+'_launch')/'request.json')
    go = contract.read_json(request['source_go_path'])
    assert argv[:5] == ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host']]
    assert go['worker'] == str(ROOT/'scripts/run_b22_worker31.py')
    assert request['spec']['execution_contract_id'] == 'C31_B22_fixed_suite_v1'
    assert request['spec']['model_limits']['predict'] == (stage['controls']+1 if actor == 'B' else 0)
    assert request['spec']['render'] is request['spec']['qualification'] is False
    assert go['inputs'][str(release_path)] == contract.identity(release_path)
