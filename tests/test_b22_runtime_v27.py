"""Versioned B22 migration and lifecycle tests; no model, GUI or physics calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('play_b22_v27_test', ROOT/'scripts/play_b22_v27.py')
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)
runtime = launcher.runtime


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return runtime.identity(path)


def dump(path, value):
    put(path, json.dumps(value).encode())


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    sources = {}
    for name in ('controller.py', 'authorized.py', 'libX11.so', 'checkpoint.zip', '__pycache__/cache.pyc'):
        path = root/name
        sources[str(path)] = put(path, ('old '+name).encode())
    base = {'schema': 'v26-test', 'source_hashes': sources, 'host': str(root/'host.py'),
            'worker': str(root/'worker.py'), 'model_limit_keys': ['load', 'torch_load', 'predict'],
            'profile_seeds': {'flat_1p6': 101}, 'mirror_seed': 102,
            'runtime_environment': {}, 'shared_session': {}}
    dump(root/runtime.BASE, base)
    monkeypatch.setattr(runtime, 'BASE_IDENTITY', runtime.identity(root/runtime.BASE))
    approved = {}
    for name in ('authorized.py', 'libX11.so'):
        path = root/name
        approved[str(path)] = {'basis': 'explicit fixture migration',
                               'identity': put(path, ('new '+name).encode())}
    monkeypatch.setattr(runtime, 'rules', lambda root=root: approved)
    for name in runtime.NEW_SOURCES:
        put(root/name, b'# New, explicitly selected v27 entry.\n')
    log = root/'dpkg.log'
    put(log, (runtime.UPGRADE_LINE+'\n').encode())
    evidence = {'package': runtime.PACKAGE, 'version': runtime.PACKAGE_VERSION,
                'installed_md5': runtime.PACKAGE_MD5, 'upgrade_record': runtime.UPGRADE_LINE,
                'upgrade_log': str(log)}
    monkeypatch.setattr(runtime, 'package_evidence', lambda: evidence)
    return root, base, approved


def test_explicit_migration_excludes_bytecode_and_pins_all_other_inputs(tree):
    root, base, approved = tree
    (root/'__pycache__/cache.pyc').unlink()
    report = runtime.build(root)
    assert report['status'] == 'READY'
    config, check = runtime.inspect(root)
    assert check['expected_bytes'] == check['verified_bytes']
    assert check['model_loads'] == check['gui_starts'] == check['physics_steps'] == 0
    assert check['gui_revalidated'] is False and check['qualification_changed'] is False
    assert config['source_hashes'][str(root/'controller.py')] == base['source_hashes'][str(root/'controller.py')]
    assert config['source_hashes'][str(root/'authorized.py')] == approved[str(root/'authorized.py')]['identity']
    assert str(root/'__pycache__/cache.pyc') not in config['source_hashes']
    assert all(str(root/name) in config['source_hashes'] for name in runtime.NEW_SOURCES)
    assert runtime.read_base(root) == base
    before = (root/runtime.MANIFEST).read_bytes()
    with pytest.raises(FileExistsError):
        runtime.build(root)
    assert (root/runtime.MANIFEST).read_bytes() == before


@pytest.mark.parametrize('name', ('controller.py', 'checkpoint.zip', 'authorized.py', 'libX11.so'))
def test_unknown_source_weight_or_library_changes_block_registration(tree, name):
    root, _, _ = tree
    put(root/name, b'not an approved identity')
    with pytest.raises(ValueError, match='unknown input differences'):
        runtime.build(root)
    assert not (root/runtime.MANIFEST).exists()


def test_registered_new_launcher_changes_are_detected(tree):
    root, _, _ = tree
    runtime.build(root)
    changed = root/runtime.NEW_SOURCES[1]
    put(changed, b'# changed after registration\n')
    _, report = runtime.inspect(root)
    assert report['status'] == 'BLOCKED'
    assert [row['path'] for row in report['unexpected_differences']] == [str(changed)]


def test_editing_new_manifest_cannot_legitimize_unexplained_weight_changes(tree):
    root, _, _ = tree
    runtime.build(root)
    manifest = json.loads((root/runtime.MANIFEST).read_text())
    weight = root/'checkpoint.zip'
    manifest['source_hashes'][str(weight)] = put(weight, b'other weights')
    with pytest.raises(ValueError, match='outside the migration rules'):
        runtime.inspect(root, manifest)


def test_unknown_closure_entries_and_runtime_configuration_changes_are_rejected(tree):
    root, _, _ = tree
    runtime.build(root)
    config, _ = runtime.inspect(root)
    for field in ('worker', 'source_hashes'):
        changed = copy.deepcopy(config)
        if field == 'worker':
            changed[field] = str(root/'different_worker.py')
        else:
            changed[field][str(root/'additional.py')] = put(root/'additional.py', b'new input')
        with pytest.raises(ValueError):
            runtime.inspect(root, changed)


def test_changed_base_and_unverified_package_cannot_be_accepted(tree, monkeypatch):
    root, _, _ = tree
    runtime.build(root)
    def bad_package():
        raise ValueError('installed package checksum differs')
    monkeypatch.setattr(runtime, 'package_evidence', bad_package)
    assert runtime.inspect(root)[1]['status'] == 'BLOCKED'
    put(root/runtime.BASE, b'{}')
    with pytest.raises(ValueError, match='trusted v26 manifest'):
        runtime.inspect(root)


def receipts(output, spec, *, controls=1600):
    output.mkdir(parents=True, exist_ok=True)
    dump(output/'session.json', {'session': 'new fake session'})
    sid = runtime.identity(output/'session.json')
    dump(output/'preflight_ready.json', {'pid': 12345, 'monotonic_s': 10.,
         'session_sha256': sid['sha256'], 'full_source_sha256_checked': True})
    dump(output/'host_receipt.json', {'execution_started_monotonic': 10., 'failure': None,
         'exit_code': 0, 'source_mismatches': [], 'owned_no_orphans': True,
         'reservation_closed': True, 'reserved_controls': spec['control_limit']})
    dump(output/'supervisor_receipt.json', {'failure': None, 'exit_code': 0,
         'cleanup': {'remaining': {}}})
    dump(output/'worker_receipt.json', {'session_identity': sid, 'failure': None,
         'archive_failed': False, 'execution_contract_id': spec['execution_contract_id'],
         'result': {'completed_controls': controls, 'python': {
             'control_completed': controls, 'native_returned': 5*controls,
             'native_attempted': 5*controls, 'native_failed': 0, 'forbidden_entries': 0}}})


def test_old_host_zero_exit_without_readiness_reproduces_false_success_and_v27_rejects_it(tmp_path, monkeypatch):
    path = ROOT/'runtime/d1_b22/research/continuation23/root23/host23.py'
    spec = importlib.util.spec_from_file_location('old_b22_host_zero_physics', path)
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    monkeypatch.setattr(host, 'subreaper', lambda: None)
    monkeypatch.setattr(host, 'clean_descendants', lambda *args: {'remaining': {}})
    process = SimpleNamespace(pid=987654, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(host.subprocess, 'Popen', lambda *args, **kwargs: process)
    output = tmp_path/'output'
    arm_spec = {'output_directory': str(output), 'render': False, 'control_limit': 1600,
                'model_limits': {}, 'soft_s': 180, 'hard_s': 240,
                'mode': 'script', 'execution_contract_id': 'C26_B22_user_session_v1'}
    source = tmp_path/'dummy_worker.py'
    source_id = put(source, b'# Dummy exits zero without any calls.\n')
    go = tmp_path/'go.json'
    dump(go, {'decision': 'GO', 'repository': str(tmp_path), 'worker': str(source),
              'inputs': {str(source): source_id}, 'runtime_environment': {},
              'shared_session': {}, 'arms': {'test': arm_spec}, 'x11_events_by_arm': {}})
    request = tmp_path/'request.json'
    dump(request, {'source_go_path': str(go), 'arm': 'test', 'spec': arm_spec})
    assert host.execute(request) == 0  # The real old control flow's false success.
    old = json.loads((output/'host_receipt.json').read_text())
    assert old['execution_started_monotonic'] is None and old['failure'] is None
    result = launcher.completion(output, arm_spec, 0)
    assert result['status'] == 'FAIL' and result['observed_counts']['completed_controls'] is None


@pytest.mark.parametrize('fault', ('no_ready', 'no_execution', 'partial', 'session', 'worker_failure',
                                 'archive_failure', 'missing_counts', 'native_count', 'host_started',
                                 'orphan', 'exit7'))
def test_exit_zero_or_complete_looking_receipts_cannot_hide_failed_execution(tmp_path, fault):
    spec = {'mode': 'script', 'control_limit': 1600,
            'execution_contract_id': 'C26_B22_user_session_v1'}
    receipts(tmp_path, spec)
    name, mutate = 'worker_receipt.json', None
    if fault == 'no_ready':
        (tmp_path/'preflight_ready.json').unlink()
    elif fault == 'no_execution':
        receipts(tmp_path, spec, controls=0)
    elif fault == 'partial':
        receipts(tmp_path, spec, controls=100)
    elif fault == 'session':
        put(tmp_path/'session.json', b'changed session')
    elif fault == 'worker_failure':
        mutate = lambda value: value.update(failure={'type': 'OwnerFailure'})
    elif fault == 'archive_failure':
        mutate = lambda value: value.update(archive_failed=True)
    elif fault == 'missing_counts':
        mutate = lambda value: value['result'].update(python=None)
    elif fault == 'native_count':
        mutate = lambda value: value['result']['python'].update(native_returned=7999)
    elif fault == 'host_started':
        name, mutate = 'host_receipt.json', lambda value: value.update(execution_started_monotonic=None)
    elif fault == 'orphan':
        name, mutate = 'supervisor_receipt.json', lambda value: value.update(cleanup={'remaining': {'2': 3}})
    if mutate:
        value = json.loads((tmp_path/name).read_text())
        mutate(value)
        dump(tmp_path/name, value)
    result = launcher.completion(tmp_path, spec, 7 if fault == 'exit7' else 0)
    assert result['status'] == 'FAIL' and result['qualification_granted'] is False
    if fault == 'missing_counts':
        assert result['observed_counts']['native_returned'] is None


def test_valid_script_or_bounded_keyboard_receipts_pass_without_granting_qualification(tmp_path):
    spec = {'mode': 'script', 'control_limit': 1600,
            'execution_contract_id': 'C26_B22_user_session_v1'}
    receipts(tmp_path, spec)
    assert launcher.completion(tmp_path, spec, 0)['status'] == 'PASS'
    spec['mode'] = 'keyboard'
    receipts(tmp_path, spec, controls=12)
    assert launcher.completion(tmp_path, spec, 0)['status'] == 'PASS'


def test_check_does_not_spawn_or_reserve_output(tree, tmp_path, monkeypatch, capsys):
    root, _, _ = tree
    runtime.build(root)
    config, check = runtime.inspect(root)
    monkeypatch.setattr(runtime, 'inspect', lambda: (config, check))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs: pytest.fail('no spawn'))
    output = tmp_path/'requested'
    assert launcher.main(['--check', '--output', str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'READY' and report['physics_steps'] == 0
    assert not output.exists() and not output.with_name(output.name+'_launch').exists()


@pytest.mark.parametrize('fault', (None, 'version', 'package_checksum', 'library_contents'))
def test_package_probe_requires_exact_version_and_installed_checksum(tmp_path, monkeypatch, fault):
    library, listing = tmp_path/'libX11.so', tmp_path/'md5sums'
    put(library, b'approved package library')
    digest = hashlib.md5(library.read_bytes()).hexdigest()
    listed = '0'*32 if fault == 'package_checksum' else digest
    put(listing, f'{listed}  {str(library).lstrip("/")}\n'.encode())
    real_path = Path
    monkeypatch.setattr(runtime, 'Path', lambda value: listing if str(value) ==
                        '/var/lib/dpkg/info/libx11-6:amd64.md5sums' else real_path(value))
    monkeypatch.setattr(runtime, 'LIBRARY', str(library))
    monkeypatch.setattr(runtime, 'PACKAGE_MD5', digest)
    version = '2:1.7.5-1ubuntu0.3' if fault == 'version' else runtime.PACKAGE_VERSION
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=version))
    if fault == 'library_contents':
        put(library, b'unapproved contents')
    if fault:
        with pytest.raises(ValueError, match='package version or installed package checksum'):
            runtime.package_evidence()
    else:
        assert runtime.package_evidence()['installed_md5'] == digest


@pytest.mark.parametrize('worker_runs', (True, False))
def test_launcher_fake_end_to_end_requires_execution_and_writes_a_bound_result(
        tree, tmp_path, monkeypatch, worker_runs):
    root, _, _ = tree
    runtime.build(root)
    config, check = runtime.inspect(root)
    monkeypatch.setattr(launcher, 'ROOT', root)
    monkeypatch.setattr(runtime, 'inspect', lambda: (config, check))
    output = tmp_path/'new_session'
    invocations = []

    def fake_host(command, **kwargs):
        invocations.append(command)
        request = json.loads(Path(command[-1]).read_text())
        go = json.loads(Path(request['source_go_path']).read_text())
        assert go['inputs'][str(root/runtime.MANIFEST)] == runtime.identity(root/runtime.MANIFEST)
        assert all(str(root/name) in go['inputs'] for name in runtime.NEW_SOURCES)
        if worker_runs:
            receipts(output, request['spec'], controls=request['spec']['control_limit'])
        else:
            output.mkdir()
            dump(output/'session.json', {'dummy': 'did not execute'})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, 'run', fake_host)
    result = launcher.main(['--headless', '--actor', 'zero', '--output', str(output)])
    assert result == (0 if worker_runs else 1) and len(invocations) == 1
    receipt = json.loads(output.with_name(output.name+'_launch').joinpath('v27_completion.json').read_text())
    assert receipt['status'] == ('PASS' if worker_runs else 'FAIL')
    assert receipt['qualification_granted'] is False
    with pytest.raises(SystemExit):
        launcher.main(['--headless', '--output', str(output)])
