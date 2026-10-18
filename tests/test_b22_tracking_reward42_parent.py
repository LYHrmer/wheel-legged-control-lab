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
    ready, rows = admit([4 << 30]*3)
    assert ready and len(rows) == 3 and rows[-1]['elapsed_s'] == 10


def test_low_samples_reset_continuous_window():
    ready, rows = admit([4 << 30]*2+[3 << 30]+[4 << 30]*3)
    assert ready and rows[-1]['elapsed_s'] == 25
    assert rows[2]['stable_elapsed_s'] is None


def test_isolated_peaks_do_not_admit():
    ready, rows = admit(([5 << 30]+[3 << 30])*30+[5 << 30])
    assert not ready and len(rows) == 61 and rows[-1]['elapsed_s'] == 300


@pytest.mark.parametrize('value', [None, True, 2*(1 << 30)-1, -1])
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
    assert plan['resource_revision']=='owned_cgroup_v5'
    assert plan['output'].endswith('_05') and p.RELEASE.name.endswith('_05.json')
    assert p.ADMISSIONS.name.endswith('_05')
    assert plan['resource_policy']['unit']=='wheel-b22-tracking42-owned05-20261018-01.service'
    assert plan['historical_actual_counts'] is None
    assert plan['historical_compiler_native_separate_from_new_budget']==6
    assert p.LIMITS['stable_effective_bytes']==p.LIMITS['start_effective_bytes']==4<<30
    assert p.LIMITS['stable_seconds']==10 and p.LIMITS['admission_minimum_samples']==3
    assert p.LIMITS['reserve_effective_bytes']==2<<30
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
    snapshot={'effective_available_bytes':4<<30,'owned':{'max_bytes':3<<30,'current_bytes':2<<30}}
    guard=SimpleNamespace(snapshot_memory=lambda scope:(calls.append(scope) or snapshot),
                          check_memory=lambda value:calls.append(value))
    monkeypatch.setattr(p,'resource_guard_api42',lambda:guard)
    assert p.validate_worker_launch42(run/'session.json',run)==session
    assert calls==[p.RESOURCE_POLICY['owned_scope'],snapshot]


@pytest.mark.parametrize('available',[None,True,(2<<30)-1])
def test_external_runtime_reserve_still_rejects_unknown_and_low(tmp_path,monkeypatch,available):
    monkeypatch.setattr(p,'resource_guard_api42',lambda:SimpleNamespace(check_memory=lambda value:None))
    with pytest.raises(ValueError,match='runtime reserve or unknown'):
        p.check_resource_snapshot42({'effective_available_bytes':available})


@pytest.fixture
def predecessor_case(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    output=tmp_path/'results/b22_tracking_reward42_20261018_04'
    release=tmp_path/'results/b22_tracking_reward42_plan_20261018_04.json'
    monkeypatch.setattr(p, 'PREDECESSOR_OUTPUT', output)
    monkeypatch.setattr(p, 'PREDECESSOR_RELEASE', release)
    run=output/'train_control';run.mkdir(parents=True)
    boot=output/'bootstrap';boot.mkdir()
    known={'controls': 0, 'normal_native': 0, 'compiler_native': 2}
    older={}
    for revision in ('02', '03'):
        path=tmp_path/f'results/b22_tracking_reward42_20261018_{revision}/report.json'
        path.parent.mkdir()
        p.write_json(path, {'status': 'INCOMPLETE', 'actual_counts': None, 'known_actual_counts': known})
        older[str(path.relative_to(tmp_path))]=p.identity(path)
    earlier=tmp_path/'results/b22_tracking_reward42_20261018_02/report.json'
    monkeypatch.setattr(p, 'EARLIER_FAILURE_REPORT', earlier)
    monkeypatch.setattr(p, 'OLDER_FAILURE_INPUTS', older)
    worker_id={'bytes': 17, 'sha256': 'a'*64}
    source={'git_commit': p.PREDECESSOR_COMMIT, 'authored': {str(p.WORKER): worker_id}}
    p.write_json(release, {'schema': 'b22-tracking42-release-v1', 'execution_contract_id': p.CONTRACT,
        'source': source, 'plan': {'output': str(output), 'limits': {k: p.LIMITS[k] for k in p.COUNT_KEYS}}})
    failure={'type': 'ValueError', 'message': p.PREDECESSOR_ERROR}
    p.write_json(run/'session.json', {'execution_contract_id': p.CONTRACT, 'name': 'train_control',
        'mode': 'train', 'reward_arm': 'control', 'alpha': 1, 'execution_output': str(output),
        'release_identity': p.identity(release), 'worker_identity': worker_id})
    p.write_json(run/'supervisor_receipt.json', {'stage': 'train_control',
        'release_identity': p.identity(release), 'session_identity': p.identity(run/'session.json'),
        'worker_identity': worker_id, 'failure': failure, 'exit_code': None,
        'supervision_returned': False, 'cleanup': {'remaining': {}}})
    p.write_json(boot/'session.json', {})
    p.write_json(boot/'supervisor_receipt.json', {})
    p.write_json(boot/'worker_receipt.json', {'status': 'COMPLETE', 'bootstrap_verified42': True,
        'session_identity': p.identity(boot/'session.json'), 'actual_counts': dict.fromkeys(p.COUNT_KEYS, 0)})
    p.write_json(output/'report.json', {'status': 'INCOMPLETE', 'actual_counts': None,
        'execution_started': True, 'active_stage': 'train_control', 'execution_contract_id': p.CONTRACT,
        'failure': failure, 'source': source, 'release_identity': p.identity(release),
        'known_actual_counts': dict.fromkeys(p.COUNT_KEYS, 0), 'cleanup': {'remaining': {}},
        'stages': [{'name': 'bootstrap', 'path': 'bootstrap/worker_receipt.json',
            'report': p.identity(boot/'worker_receipt.json'), 'independent_saved_reader_executed': False}],
        'predecessor_failure': {'inputs': {str(tmp_path/k): v for k,v in older.items()},
            'historical_actual_counts': None, 'historical_known_counts': known,
            'historical_compiler_native_separate_from_new_budget': 4,
            'model_load_or_learning_occurred': False,
            'earlier02_failure': {'historical_actual_counts': None, 'historical_known_counts': known,
                                 'identity': p.identity(earlier)}}})
    progress={'completed_controls': 26624,
        'rollout': {'rollout_index': 25, 'transitions': 1024, 'num_timesteps_at_end': 26624},
        'boundary': {'C_state': {'construction_attempts': 2, 'construction_returns': 2,
            'control_attempts': 133120, 'control_returns': 133120}, 'python': {
            'control_attempted': 26624, 'control_completed': 26624, 'native_attempted': 133120,
            'native_returned': 133120, 'clock_advanced_substeps': 133120}}}
    (run/'training').mkdir()
    p.write_json(run/'training/progress_rollout_0025.json', progress)
    p.write_json(run/'failure_learning_receipt.json', {'policy_state_changed': True,
        'optimizer_state_changed': True, 'actor_mean_state_changed': True,
        'actual': {'optimizer_steps': 387, 'transitions': 27000},
        'construction': {'parent_final_zip_sha256': '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'}})
    p.write_json(run/'training/training_segment_receipt.json', {'native_returned': 135000})
    payloads=('training/progress_rollout_0025.json', 'failure_learning_receipt.json',
              'training/training_segment_receipt.json')
    def repin():
        for name in payloads:
            path=run/name
            p.write_json(path.with_name(path.name+'.manifest.json'), {
                'schema': 'd1-archive-transaction-13-v1',
                'payloads': [{'file': path.name, **p.identity(path)}]}, replace=path.with_name(path.name+'.manifest.json').exists())
        files=[release, output/'report.json', run/'session.json', run/'supervisor_receipt.json',
               *[boot/name for name in ('session.json','supervisor_receipt.json','worker_receipt.json')],
               *[run/name for name in payloads],
               *[(run/name).with_name((run/name).name+'.manifest.json') for name in payloads]]
        monkeypatch.setattr(p, 'PREDECESSOR_INPUTS', {
            **older, **{str(path.relative_to(tmp_path)): p.identity(path) for path in files}})
    def mutate(name, keys, value, *, update_pin=True):
        path=output/name;data=p.read_json(path);cursor=data
        for key in keys[:-1]:cursor=cursor[key]
        cursor[keys[-1]]=value
        p.write_json(path, data, replace=True)
        if update_pin:repin()
    repin()
    return mutate


def test_trained_predecessor_is_not_zero_or_unused_and_cannot_be_resumed(predecessor_case):
    proof=p.predecessor_failure42()
    assert proof['historical_actual_counts'] is None
    assert proof['parent_reported_known_counts']==dict.fromkeys(p.COUNT_KEYS,0)
    assert proof['parent_zero_known_does_not_mean_zero_work'] is True
    assert proof['model_load_or_learning_occurred'] is True
    assert proof['persisted_rollout_lower_bound']=={'controls':26624,'normal_native':133120,'compiler_native':2}
    assert proof['additional_reported_lower_bound']=={'controls':27000,'normal_native':135000,'compiler_native':2}
    assert proof['earlier02_03_known_compiler_native']==4
    assert proof['historical_compiler_native_separate_from_new_budget']==6
    assert proof['partial_checkpoint_reused'] is False
    assert proof['new_attempt_is_independent_budget'] is True


@pytest.mark.parametrize(('file','keys','value'),[
    ('report.json',['actual_counts'],{'controls':0,'normal_native':0,'compiler_native':0}),
    ('report.json',['known_actual_counts'],None),
    ('report.json',['predecessor_failure'],None),
    ('report.json',['predecessor_failure','historical_compiler_native_separate_from_new_budget'],0),
    ('report.json',['cleanup','remaining'],{'123':4}),
    ('train_control/session.json',['worker_identity'],{'bytes':0,'sha256':'b'*64}),
    ('train_control/training/progress_rollout_0025.json',['completed_controls'],0),
    ('train_control/training/progress_rollout_0025.json',['boundary','C_state','control_returns'],0),
    ('train_control/training/progress_rollout_0025.json',['boundary','python','native_returned'],True),
    ('train_control/failure_learning_receipt.json',['policy_state_changed'],False),
    ('train_control/failure_learning_receipt.json',['actual','optimizer_steps'],0),
    ('train_control/failure_learning_receipt.json',['actual','transitions'],None),
    ('train_control/training/training_segment_receipt.json',['native_returned'],0),
])
def test_trained_history_unknown_tampered_or_missing_learning_fails(predecessor_case,file,keys,value):
    predecessor_case(file,keys,value)
    with pytest.raises(ValueError,match='predecessor'):p.predecessor_failure42()


def test_fixed_predecessor_bytes_cannot_be_rebaselined(predecessor_case):
    predecessor_case('report.json',['status'],'different',update_pin=False)
    with pytest.raises(ValueError,match='receipt identity changed'):p.predecessor_failure42()


def test_minimum_includes_sample_that_triggers_runtime_stop(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'OUTPUT',tmp_path/'unclaimed')
    monkeypatch.setattr(p,'ADMISSIONS',tmp_path/'admissions')
    monkeypatch.setattr(p,'unused_budget_origin42',dict)
    monkeypatch.setattr(p,'predecessor_failure42',dict)
    monkeypatch.setattr(p.os,'getpriority',lambda *a:10)
    monkeypatch.setattr(p,'supervision',lambda:SimpleNamespace())
    monkeypatch.setattr(p,'resource_guard_api42',lambda:SimpleNamespace(check_memory=lambda _:None))
    bad=(2<<30)-1
    monkeypatch.setattr(p,'resource_snapshot42',lambda:{'effective_available_bytes':bad})
    report=p.execute42()
    assert report['status']=='NOT_READY' and report['minimum_effective_bytes']==bad
    assert report['execution_started'] is False


def test_phase_cleanup_normal_exit_does_not_wait():
    clock=Clock();calls=[]
    host=SimpleNamespace(descendants=lambda _: {},signal_identity=lambda *a:pytest.fail('unexpected TERM'),
        clean_descendants=lambda *a:(calls.append('fallback') or {'remaining':{}}))
    record=p.phase_cleanup42(host,99,None,deadline=100,clock=clock,sleeper=clock.sleep)
    assert clock.now==0 and calls==['fallback'] and record['term_attempted_identities']==[]


def test_phase_cleanup_terms_each_owned_birth_once_including_new_children():
    clock=Clock();terms=[]
    def alive(_):
        if clock.now<.25:return {1:10}
        if clock.now<.5:return {1:10,2:20}
        if clock.now<.75:return {1:11,2:20}
        return {}
    def term(pid,birth,number):
        assert alive(99).get(pid)==birth
        terms.append((pid,birth,number))
    host=SimpleNamespace(descendants=alive,signal_identity=term,clean_descendants=lambda *a:{'remaining':{}})
    record=p.phase_cleanup42(host,99,None,deadline=100,clock=clock,sleeper=clock.sleep)
    assert [(pid,birth) for pid,birth,_ in terms]==[(1,10),(2,20),(1,11)]
    assert all(number==p.signal.SIGTERM for _,_,number in terms)
    assert record['grace_elapsed_s']==.75


@pytest.mark.parametrize(('deadline','grace'),[(100,15),(7,2)])
def test_phase_cleanup_bounded_grace_then_frozen_fallback(deadline,grace):
    clock=Clock();terms=[]
    def fallback(*args):clock.sleep(4.9);return {'remaining':{}}
    host=SimpleNamespace(descendants=lambda _: {1:10},signal_identity=lambda *a:terms.append(a),
                         clean_descendants=fallback)
    record=p.phase_cleanup42(host,99,None,deadline=deadline,clock=clock,sleeper=clock.sleep)
    assert len(terms)==1 and record['grace_elapsed_s']==grace
    assert record['elapsed_s']==pytest.approx(grace+4.9)


def test_cleanup_error_keeps_primary_failure_and_still_writes_actual_supervisor(tmp_path):
    import ast
    source=ast.parse((ROOT/'scripts/run_b22_tracking_reward42.py').read_text())
    function=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='run_process')
    clock=Clock();report={};run=tmp_path/'train_control';run.mkdir()
    def failed_cleanup(*args):raise RuntimeError('cleanup failed')
    real_host=SimpleNamespace(descendants=lambda _:{},signal_identity=lambda *a:None,
                              clean_descendants=failed_cleanup)
    def supervise(*args,host,**kwargs):
        try:raise ValueError('resource stop')
        finally:host.clean_descendants(99,None)
    namespace={'time':SimpleNamespace(monotonic=clock),'os':p.os,'sys':p.sys,'subprocess':p.subprocess,
        'SimpleNamespace':SimpleNamespace,'Path':Path,'report':report,'host':real_host,
        'common':SimpleNamespace(supervise=supervise),'sample':lambda *a:None,'deadline':100,
        'ROOT':tmp_path,'OUTPUT':tmp_path,'RELEASE':tmp_path/'release','WORKER':tmp_path/'worker',
        'require':p.require,'identity':lambda _:{'bytes':1,'sha256':'a'*64},'write_json':p.write_json,
        'phase_cleanup42':lambda *a,**kw:p.phase_cleanup42(*a,**kw,clock=clock,sleeper=clock.sleep)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])), '<actual-run-process>', 'exec'), namespace)  # noqa: S102 -- execute only the owned production seam with fake dependencies
    with pytest.raises(ValueError,match='resource stop'):
        namespace['run_process']([],tmp_path/'worker.log',100,stage={'name':'train_control'})
    receipt=p.read_json(run/'supervisor_receipt.json')
    assert receipt['failure']=={'type':'ValueError','message':'resource stop'}
    assert receipt['cleanup_error'] is not None
    assert len(receipt['phase_cleanup_records'])==2
    assert all(r['errors'][0]['message']=='cleanup failed' for r in receipt['phase_cleanup_records'])
    assert len(report['phase_cleanup_records'][0]['records'])==2
