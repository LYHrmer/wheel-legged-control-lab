"""Pure orchestration failures; no frameworks, policy calls or physical execution."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tracking42_parent_test', ROOT/'scripts/run_b22_tracking_reward42.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def admit(values, *, max_wait=300):
    clock = Clock()
    rows = []
    sequence = iter(values)
    result = p.stable_admission42(lambda: next(sequence), rows.append,
                                  clock=clock, sleeper=clock.sleep, max_wait=max_wait)
    return result, rows


def test_stable_admission_needs_three_samples_over_ten_seconds():
    ready, rows = admit([6 << 30]*3)
    assert ready and len(rows) == 3 and rows[-1]['elapsed_s'] == 10


def test_low_samples_reset_continuous_window():
    ready, rows = admit([6 << 30]*2+[5 << 30]+[6 << 30]*3)
    assert ready and rows[-1]['elapsed_s'] == 25
    assert rows[2]['stable_elapsed_s'] is None


def test_isolated_peaks_do_not_admit():
    ready, rows = admit(([7 << 30]+[5 << 30])*30+[7 << 30])
    assert not ready and len(rows) == 61 and rows[-1]['elapsed_s'] == 300


@pytest.mark.parametrize('value', [None, True, 4*(1 << 30)-1, -1])
def test_unknown_or_runtime_low_rejected(value):
    with pytest.raises(ValueError, match='admission memory'):
        admit([value])


def test_delayed_sample_cannot_exceed_wait_limit_and_admit():
    clock = Clock()
    def sample():
        clock.now += 10
        return 8 << 30
    rows = []
    assert not p.stable_admission42(sample, rows.append, clock=clock,
                                    sleeper=clock.sleep, max_wait=65)


def test_declared_budget_and_order():
    stages = p.stages42()
    assert [s['name'] for s in stages] == ['bootstrap', 'train_control', 'train_deficit8', 'eval']
    assert [s['alpha'] for s in stages] == [None, 1, 8, None]
    for key in p.COUNT_KEYS:
        assert sum(s[key] for s in stages) == p.LIMITS[key]
    assert stages[3]['controls'] == 3*(5*1600+1800)+2*600
    assert all(stages[0][key] == 0 for key in p.COUNT_KEYS)
    assert sum(s['pure_servo_advance_max'] for s in stages) == 350600
    assert sum(s['reader_pure_servo_advance_max'] for s in stages) == 190600
    assert p.describe42()['selection'].endswith('floors do not screen arms')


def test_two_high_samples_sixty_one_seconds_apart_do_not_admit():
    clock = Clock()
    rows = []
    assert not p.stable_admission42(lambda: 8 << 30, rows.append, clock=clock,
                                    sleeper=lambda _: clock.sleep(61), max_wait=300)
    assert len(rows) == 6 and rows[-1]['consecutive_qualified_samples'] == 1


def test_large_scheduler_gap_resets_but_later_three_samples_can_pass():
    clock = Clock()
    rows = []
    delays = iter([61]+[5]*2)
    assert p.stable_admission42(lambda: 8 << 30, rows.append, clock=clock,
                                sleeper=lambda _: clock.sleep(next(delays)))
    assert rows[-1]['elapsed_s'] == 71
    assert rows[-1]['consecutive_qualified_samples'] == 3


def test_slow_final_receipt_cannot_admit_after_wait_deadline():
    clock = Clock()
    rows = []
    def record(row):
        rows.append(row)
        if len(rows) == 3:
            clock.sleep(300)
    assert not p.stable_admission42(lambda: 8 << 30, record, clock=clock, sleeper=clock.sleep)
    assert clock.now == 310


@pytest.mark.parametrize('counts', [None, {}, {'controls': True, 'normal_native': 5, 'compiler_native': 2},
    {'controls': 1, 'normal_native': 4, 'compiler_native': 2},
    {'controls': 1, 'normal_native': 11, 'compiler_native': 2},
    {'controls': 1, 'normal_native': 5, 'compiler_native': 3},
    {'controls': 32769, 'normal_native': 163845, 'compiler_native': 2}])
def test_bad_partial_counts_rejected(counts):
    with pytest.raises(ValueError):
        p.checked_counts42(counts, p.stage_by_name42('train_control'), complete=False)


def test_partial_native_tick_retained_but_never_complete():
    value = {'controls': 12, 'normal_native': 63, 'compiler_native': 2}
    assert p.checked_counts42(value, p.stage_by_name42('train_control'), complete=False) == value
    with pytest.raises(ValueError):
        p.checked_counts42(value, p.stage_by_name42('train_control'), complete=True)


def test_short_train_cannot_be_final_but_legitimate_eval_failure_is_data():
    value = {'controls': 12, 'normal_native': 60, 'compiler_native': 2}
    with pytest.raises(ValueError, match='partial training'):
        p.checked_counts42(value, p.stage_by_name42('train_control'), complete=True)
    assert p.checked_counts42(value, p.stage_by_name42('eval'), complete=True) == value


def test_missing_worker_receipt_stays_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path)
    assert p.known_counts42(p.stage_by_name42('train_control')) is None


def test_foreign_partial_receipt_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path)
    run = tmp_path/'train_control'
    run.mkdir()
    (run/'session.json').write_text('{}')
    (run/'worker_receipt.json').write_text(json.dumps({
        'session_identity': {'bytes': 2, 'sha256': '0'*64}, 'execution_contract_id': p.CONTRACT,
        'known_partial_counts': {'controls': 12, 'normal_native': 60, 'compiler_native': 2}}))
    assert p.known_counts42(p.stage_by_name42('train_control')) is None


def test_conflicting_exact_and_partial_counts_cannot_suppress_known_spending(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path)
    run = tmp_path/'train_control'
    run.mkdir()
    (run/'session.json').write_text('{}')
    (run/'worker_receipt.json').write_text(json.dumps({
        'session_identity': p.identity(run/'session.json'), 'execution_contract_id': p.CONTRACT,
        'actual_counts': dict.fromkeys(p.COUNT_KEYS, 0),
        'known_partial_counts': {'controls': 5, 'normal_native': 25, 'compiler_native': 2}}))
    assert p.known_counts42(p.stage_by_name42('train_control')) is None


def test_payload_quota_counts_temporary_files_and_rejects_symlinks(tmp_path):
    (tmp_path/'partial.tmp').write_bytes(b'x'*7)
    (tmp_path/'folder').mkdir()
    (tmp_path/'folder/payload').write_bytes(b'x'*11)
    assert p.directory_bytes42(tmp_path) == 18
    (tmp_path/'linked').symlink_to(tmp_path/'folder/payload')
    with pytest.raises(ValueError, match='symlink'):
        p.directory_bytes42(tmp_path)


def test_reader_environment_cannot_inherit_engine_preload(monkeypatch):
    monkeypatch.setenv('LD_PRELOAD', '/engine.so')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/engine')
    monkeypatch.setenv('PYTHONPATH', '/execution')
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    environment = p.reader_environment42()
    assert all(key not in environment for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH'))
    assert environment['OMP_NUM_THREADS'] == environment['OPENBLAS_NUM_THREADS'] == '1'


def test_quota_snapshot_tolerates_owned_temporary_publication(tmp_path, monkeypatch):
    disappearing = tmp_path/'payload.partial'
    disappearing.write_bytes(b'x'*50)
    stable = tmp_path/'saved'
    stable.write_bytes(b'x'*12)
    original = Path.lstat
    def race(path):
        if path == disappearing:
            raise FileNotFoundError('atomic archive published and removed its temporary')
        return original(path)
    monkeypatch.setattr(Path, 'lstat', race)
    assert p.directory_bytes42(tmp_path) == 12


def test_json_unknown_survives_and_replacement_is_explicit(tmp_path):
    path = tmp_path/'report.json'
    p.write_json(path, {'actual_counts': None})
    assert p.read_json(path)['actual_counts'] is None
    with pytest.raises(FileExistsError):
        p.write_json(path, {})
    p.write_json(path, {'status': 'NOT_READY'}, replace=True)
    assert p.read_json(path) == {'status': 'NOT_READY'}
    assert not path.with_name('report.json.tmp').exists()


def test_final_receipt_can_replace_after_interrupted_atomic_save(tmp_path, monkeypatch):
    path = tmp_path/'report.json'
    p.write_json(path, {'status': 'RUNNING'})
    original = p.os.replace
    def interrupted(*_):
        raise InterruptedError('signal just before replacement')
    monkeypatch.setattr(p.os, 'replace', interrupted)
    with pytest.raises(InterruptedError):
        p.write_json(path, {'status': 'WAITING'}, replace=True)
    monkeypatch.setattr(p.os, 'replace', original)
    p.write_json(path, {'status': 'INCOMPLETE'}, replace=True)
    assert p.read_json(path) == {'status': 'INCOMPLETE'}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('mismatch', ['initial_state', 'initial_learning_state', 'first1024_numeric', 'first1024_gaussian'])
def test_pairing_rejects_different_actual_initial_or_first_rollout(tmp_path, monkeypatch, mismatch):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path)
    release = tmp_path/'release.json'
    release.write_text('{}')
    monkeypatch.setattr(p, 'RELEASE', release)
    for arm in ('control', 'deficit8'):
        pairing = {'schema': 'b22-tracking42-pairing-v1', 'seed': 201011, 'course_arm': 'B',
                   'initial_state': {'digest': 'same'}, 'initial_learning_state': {'digest': 'same'},
                   'first1024_numeric': {'digest': 'same'}, 'first1024_gaussian': {'digest': 'same'}}
        if arm == 'deficit8':
            pairing[mismatch] = {'digest': 'different'}
        p.write_json(tmp_path/('train_'+arm+'_readback.json'), {
            'status': 'COMPLETE', 'training_valid': True, 'stage': 'train_'+arm,
            'release_identity': p.identity(release), 'pairing42': pairing})
    with pytest.raises(ValueError, match='nonreward data differ'):
        p.verify_training_pair42()


def test_cold_default_imports_no_execution_stack():
    code = """
import importlib.abc, importlib.util, json, sys
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in ('numpy','torch','mujoco','stable_baselines3','gymnasium','engine_binding'):
   raise AssertionError('unexpected execution import '+name)
sys.meta_path.insert(0, Reject())
spec=importlib.util.spec_from_file_location('parent42',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.main([])==0
"""
    result = subprocess.run([sys.executable, '-B', '-c', code, str(p.DRIVER)],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['status'] == 'PLAN_ONLY' and value['execution_started'] is False


def test_execution_cannot_retry_claimed_output(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path)
    with pytest.raises(ValueError, match='already claimed'):
        p.execute42()


def test_prepare_cannot_replace_release(tmp_path, monkeypatch):
    release = tmp_path/'plan.json'
    release.write_text('{}')
    monkeypatch.setattr(p, 'RELEASE', release)
    monkeypatch.setattr(p, 'OUTPUT', tmp_path/'not_created')
    with pytest.raises(ValueError, match='already exists'):
        p.prepare42()


def test_checkpoint_evaluation_requires_both_complete_readers(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path)
    (tmp_path/'train_control_readback.json').write_text(json.dumps({
        'status': 'INCOMPLETE', 'training_valid': False}))
    with pytest.raises(ValueError, match='both complete'):
        p.eval_inputs42()


@pytest.fixture
def comparison_case(tmp_path, monkeypatch):
    release, reader = tmp_path/'release.json', tmp_path/'reader.py'
    release.write_text('{}')
    reader.write_text('# saved-only reader')
    monkeypatch.setattr(p, 'RELEASE', release)
    monkeypatch.setattr(p, 'READER', reader)
    stages = [{'name': 'bootstrap', 'path': 'bootstrap/worker_receipt.json',
               'report': {'sha256': 'bootstrap', 'bytes': 9},
               'independent_saved_reader_executed': False}]
    stages += [{'name': name, 'report': {'sha256': name, 'bytes': 8}}
               for name in ('train_control', 'train_deficit8', 'eval')]
    value = {'schema': 'b22-tracking42-comparison-v1', 'status': 'COMPLETE',
             'execution_contract_id': p.CONTRACT, 'release_identity': p.identity(release),
             'reader_identity': p.identity(reader),
             'stage_reports': {row['name']: row['report'] for row in stages[1:]},
             'reader_model_calls': 0, 'reader_physics_calls': 0, 'actual_pure_servo_advances': 0}
    return value, stages


def test_comparison_accepts_three_readbacks_with_separate_bootstrap_receipt(comparison_case):
    p.validate_comparison42(*comparison_case)


def test_comparison_rejects_changed_previously_accepted_readback(comparison_case):
    value, stages = comparison_case
    value['stage_reports']['train_control'] = {'sha256': 'changed', 'bytes': 8}
    with pytest.raises(ValueError, match='three accepted numerical'):
        p.validate_comparison42(value, stages)


@pytest.mark.parametrize('field', ['reader_model_calls', 'reader_physics_calls', 'actual_pure_servo_advances'])
def test_comparison_rejects_unregistered_execution(comparison_case, field):
    value, stages = comparison_case
    value[field] = 1
    with pytest.raises(ValueError, match='saved-summary mathematics'):
        p.validate_comparison42(value, stages)


def test_comparison_requires_current_reader_identity(comparison_case):
    value, stages = comparison_case
    value['reader_identity'] = {'sha256': 'different', 'bytes': 8}
    with pytest.raises(ValueError, match='provenance'):
        p.validate_comparison42(value, stages)


def test_resource_revision_keeps_science_and_declares_one_owned_job():
    plan=p.describe42()
    assert plan['resource_revision']=='owned_cgroup_v4'
    assert plan['output'].endswith('_04') and p.RELEASE.name.endswith('_04.json')
    assert p.ADMISSIONS.name.endswith('_04')
    assert plan['resource_policy']['unit']=='wheel-b22-tracking42-owned04-20261018-01.service'
    assert plan['historical_actual_counts'] is None
    assert plan['historical_compiler_native_separate_from_new_budget']==4
    assert p.LIMITS['stable_effective_bytes']==p.LIMITS['start_effective_bytes']==6<<30
    assert p.LIMITS['stable_seconds']==10 and p.LIMITS['admission_minimum_samples']==3
    assert p.LIMITS['reserve_effective_bytes']==4<<30
    assert plan['resource_policy']['memory_high_bytes']==2<<30
    assert plan['resource_policy']['memory_max_bytes']==3<<30
    assert plan['resource_policy']['memory_swap_max_bytes']==0
    assert plan['resource_policy']['memory_oom_group']==1
    assert plan['resource_policy']['oom_policy']=='kill'
    assert plan['resource_policy']['owned_scope'].endswith('/'+plan['resource_policy']['unit'])
    assert 'saved readers' in plan['resource_policy']['covers']


def test_original_prepared_budget_must_be_exact_and_unclaimed(tmp_path,monkeypatch):
    old_release=tmp_path/'old_release.json';old_output=tmp_path/'old_output'
    monkeypatch.setattr(p,'UNUSED_RELEASE',old_release)
    monkeypatch.setattr(p,'UNUSED_OUTPUT',old_output)
    value={'execution_contract_id':p.CONTRACT,'source':{'git_commit':p.UNUSED_COMMIT},
        'plan':{'output':str(old_output),'limits':{k:p.LIMITS[k] for k in p.COUNT_KEYS}}}
    p.write_json(old_release,value)
    monkeypatch.setattr(p,'UNUSED_RELEASE_SHA',p.identity(old_release)['sha256'])
    assert p.unused_budget_origin42()['original_scientific_budget_origin_only']
    old_output.mkdir()
    with pytest.raises(ValueError,match='previously claimed'):p.unused_budget_origin42()
    old_output.rmdir()
    old_release.write_text('{}')
    with pytest.raises(ValueError,match='release changed'):p.unused_budget_origin42()


def test_worker_gate_uses_external_headroom_under_own_three_gib_cap(tmp_path,monkeypatch):
    output=tmp_path/'execution';run=output/'bootstrap';run.mkdir(parents=True)
    release=tmp_path/'release.json';release.write_text('{}')
    argv=[str(p.WORKER),'--session',str(run/'session.json'),'--output',str(run),'--bootstrap']
    session={'release_path':str(release),'name':'bootstrap','parent_pid':777,'argv':argv,
        'nonce':'a'*48,'host_started_monotonic':p.time.monotonic(),
        'wall_clock_utc':'2026-10-18T00:00:00+00:00','runtime_environment':{},'soft_s':240}
    p.write_json(run/'session.json',session)
    p.write_json(output/'bootstrap_reservation.json',{
        'nonce':session['nonce'],'session_identity':p.identity(run/'session.json')})
    monkeypatch.setattr(p,'OUTPUT',output)
    monkeypatch.setattr(p,'validate_release42',lambda _: {})
    monkeypatch.setattr(p,'make_session42',lambda *args,**kwargs:dict(session))
    monkeypatch.setattr(p.os,'getppid',lambda:777)
    monkeypatch.setattr(p.os,'getpriority',lambda *args:10)
    monkeypatch.setattr(p.sys,'argv',argv)
    monkeypatch.setattr(p,'supervision',lambda:pytest.fail('old min-own-cap RAM helper must not be used'))
    calls=[]
    snapshot={'effective_available_bytes':6<<30,'owned':{'max_bytes':3<<30,'current_bytes':2<<30}}
    guard=SimpleNamespace(snapshot_memory=lambda scope:(calls.append(scope) or snapshot),
                          check_memory=lambda value:calls.append(value))
    monkeypatch.setattr(p,'resource_guard_api42',lambda:guard)
    assert p.validate_worker_launch42(run/'session.json',run)==session
    assert calls==[p.RESOURCE_POLICY['owned_scope'],snapshot]


@pytest.mark.parametrize('available',[None,True,(4<<30)-1])
def test_external_runtime_reserve_still_rejects_unknown_and_low(tmp_path,monkeypatch,available):
    monkeypatch.setattr(p,'resource_guard_api42',lambda:SimpleNamespace(check_memory=lambda value:None))
    with pytest.raises(ValueError,match='runtime reserve or unknown'):
        p.check_resource_snapshot42({'effective_available_bytes':available})


@pytest.fixture
def predecessor_case(tmp_path, monkeypatch):
    """Synthetic closed03 receipts retain their known02 failure, without models."""
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    output=tmp_path/'results/b22_tracking_reward42_20261018_03'
    release=tmp_path/'results/b22_tracking_reward42_plan_20261018_03.json'
    monkeypatch.setattr(p, 'PREDECESSOR_OUTPUT', output)
    monkeypatch.setattr(p, 'PREDECESSOR_RELEASE', release)
    (output/'bootstrap').mkdir(parents=True)
    (output/'train_control').mkdir()
    old_worker={'bytes': 17, 'sha256': 'a'*64}
    source={'git_commit': p.PREDECESSOR_COMMIT, 'authored': {str(p.WORKER): old_worker}}
    p.write_json(release, {'schema': 'b22-tracking42-release-v1',
        'execution_contract_id': p.CONTRACT, 'source': source,
        'plan': {'output': str(output), 'limits': {k: p.LIMITS[k] for k in p.COUNT_KEYS}}})
    release_id=p.identity(release)
    known={'controls': 0, 'normal_native': 0, 'compiler_native': 2}
    earlier=tmp_path/'results/b22_tracking_reward42_20261018_02/report.json'
    earlier.parent.mkdir()
    p.write_json(earlier, {'execution_contract_id': p.CONTRACT, 'execution_started': True,
        'status': 'INCOMPLETE', 'actual_counts': None, 'known_actual_counts': known,
        'source': {'git_commit': p.EARLIER_FAILURE_COMMIT}, 'cleanup': {'remaining': {}}})
    monkeypatch.setattr(p, 'EARLIER_FAILURE_REPORT', earlier)
    monkeypatch.setattr(p, 'EARLIER_FAILURE_IDENTITY', p.identity(earlier))
    for stage, code in (('bootstrap', 0), ('train_control', 1)):
        run=output/stage
        p.write_json(run/'session.json', {'execution_contract_id': p.CONTRACT, 'name': stage,
            'mode': 'bootstrap' if stage=='bootstrap' else 'train', 'release_path': str(release),
            'release_identity': release_id, 'execution_output': str(output), 'worker_identity': old_worker})
        worker={'schema': 'b22-tracking42-worker-receipt-v1', 'execution_contract_id': p.CONTRACT,
            'session_identity': p.identity(run/'session.json'), 'cleanup_errors': [],
            'model_calls': {'schema': 'd1-model-api-ledger-20-v1', 'counts': {}, 'actor_rows': 0, 'critic_rows': 0},
            'actual_pure_servo_advances': 0}
        if stage=='bootstrap':
            worker.update(status='COMPLETE', failure=None, bootstrap_verified42=True,
                          actual_counts=dict.fromkeys(p.COUNT_KEYS, 0))
        else:
            worker.update(status='INCOMPLETE', execution_complete=False, actual_counts=None,
                known_partial_counts=known, retry_permitted=False,
                failure={'type': 'FileNotFoundError', 'message': p.PREDECESSOR_ERROR,
                    'training_cleanup_errors': [],
                    'traceback': 'File "runtime_support22.py", line 89, in atomic_training_io\n'
                        'import curriculum22 as module\nFile "recipes22.py"\nFileNotFoundError: '+p.PREDECESSOR_ERROR+'\n'},
                C_final={'construction_attempts': 2, 'construction_returns': 2,
                    'control_attempts': 0, 'control_returns': 0, 'ccd_attempts': 0, 'ccd_returns': 0, 'violations': 0},
                python=dict.fromkeys(('control_attempted', 'control_completed', 'native_attempted',
                    'native_returned', 'native_failed', 'clock_advanced_substeps'), 0))
        p.write_json(run/'worker_receipt.json', worker)
        p.write_json(run/'supervisor_receipt.json', {
            'schema': 'b22-tracking42-worker-supervision-v1', 'stage': stage,
            'release_identity': release_id, 'session_identity': p.identity(run/'session.json'),
            'worker_identity': old_worker, 'exit_code': code, 'failure': None,
            'supervision_returned': True, 'cleanup': {'remaining': {}, 'signalled_identities': {}}})
    p.write_json(output/'report.json', {'schema': 'b22-tracking42-supervision-v1',
        'execution_contract_id': p.CONTRACT, 'status': 'INCOMPLETE', 'execution_started': True,
        'actual_counts': None, 'known_actual_counts': known, 'active_stage': 'train_control',
        'retry_permitted': False, 'release_identity': release_id, 'source': source,
        'failure': {'type': 'ValueError', 'message': 'C42 worker did not complete with clean ownership'},
        'predecessor_failure': {'schema': 'b22-tracking42-specific-predecessor-failure-v1',
            'source_commit': p.EARLIER_FAILURE_COMMIT, 'historical_attempt_occurred': True,
            'historical_actual_counts': None, 'historical_known_counts': known,
            'historical_compiler_native_separate_from_new_budget': 2,
            'model_load_or_learning_occurred': False, 'clean_process_closure': True,
            'failure': {'type': 'TypeError', 'message': 'C18 installer needs the exact production world-upright env'},
            'inputs': {str(earlier): p.EARLIER_FAILURE_IDENTITY}},
        'cleanup': {'remaining': {}, 'signalled_identities': {}},
        'stages': [{'name': 'bootstrap', 'path': 'bootstrap/worker_receipt.json',
            'report': p.identity(output/'bootstrap/worker_receipt.json'),
            'independent_saved_reader_executed': False}]})
    def repin():
        monkeypatch.setattr(p, 'PREDECESSOR_INPUTS', {
            str(path.relative_to(tmp_path)): p.identity(path)
            for path in [release, output/'report.json', earlier,
                *[output/stage/name for stage in ('bootstrap', 'train_control')
                  for name in ('session.json', 'worker_receipt.json', 'supervisor_receipt.json')]]})
    def mutate(name, keys, value, *, update_pin=True):
        path=output/name;data=p.read_json(path);cursor=data
        for key in keys[:-1]:cursor=cursor[key]
        cursor[keys[-1]]=value
        p.write_json(path, data, replace=True)
        if update_pin:repin()
    repin()
    return mutate


def test_specific_predecessor_preserves_unknown_total_and_separate_compiler_spend(predecessor_case):
    proof=p.predecessor_failure42()
    assert proof['historical_attempt_occurred'] is True
    assert proof['historical_actual_counts'] is None
    assert proof['historical_known_counts']=={'controls': 0, 'normal_native': 0, 'compiler_native': 2}
    assert proof['historical_compiler_native_separate_from_new_budget']==4
    assert proof['earlier02_failure']['historical_actual_counts'] is None
    assert proof['earlier02_failure']['historical_known_counts']['compiler_native']==2
    assert proof['model_load_or_learning_occurred'] is False
    assert proof['clean_process_closure'] is True
    assert len(proof['inputs'])==9


@pytest.mark.parametrize(('file','keys','value'), [
    ('train_control/worker_receipt.json', ['known_partial_counts'], None),
    ('train_control/worker_receipt.json', ['known_partial_counts','controls'], 1),
    ('train_control/worker_receipt.json', ['known_partial_counts','controls'], False),
    ('train_control/worker_receipt.json', ['known_partial_counts','normal_native'], 1),
    ('train_control/worker_receipt.json', ['known_partial_counts','compiler_native'], 0),
    ('train_control/worker_receipt.json', ['model_calls','counts'], {'learn': 1}),
    ('train_control/worker_receipt.json', ['model_calls','counts'], {'load': 1}),
    ('train_control/worker_receipt.json', ['cleanup_errors'], ['close failed']),
    ('train_control/worker_receipt.json', ['failure','message'], 'other TypeError'),
    ('train_control/worker_receipt.json', ['failure','type'], 'TypeError'),
    ('train_control/worker_receipt.json', ['failure','traceback'], 'FileNotFoundError: '+p.PREDECESSOR_ERROR+'\n'),
    ('train_control/worker_receipt.json', ['failure','training_cleanup_errors'], ['dirty']),
    ('train_control/worker_receipt.json', ['C_final','construction_returns'], 1),
    ('train_control/supervisor_receipt.json', ['cleanup','remaining'], {'123': 4}),
    ('train_control/supervisor_receipt.json', ['exit_code'], None),
    ('train_control/supervisor_receipt.json', ['session_identity'], {'bytes': 0, 'sha256': 'b'*64}),
    ('report.json', ['actual_counts'], {'controls': 0, 'normal_native': 0, 'compiler_native': 2}),
    ('report.json', ['known_actual_counts','controls'], 1),
    ('report.json', ['cleanup','remaining'], {'123': 4}),
    ('report.json', ['predecessor_failure'], None),
    ('report.json', ['predecessor_failure','historical_known_counts','compiler_native'], 0),
    ('report.json', ['predecessor_failure','model_load_or_learning_occurred'], True),
    ('report.json', ['predecessor_failure','clean_process_closure'], False),
])
def test_predecessor_cannot_hide_unknown_learning_spend_or_dirty_exit(predecessor_case, file, keys, value):
    predecessor_case(file, keys, value)
    with pytest.raises(ValueError, match='predecessor'):
        p.predecessor_failure42()


def test_fixed_predecessor_bytes_cannot_be_rebaselined(predecessor_case):
    predecessor_case('train_control/worker_receipt.json', ['status'], 'different', update_pin=False)
    with pytest.raises(ValueError, match='receipt identity changed'):
        p.predecessor_failure42()
