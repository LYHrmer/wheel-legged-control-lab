"""Pure orchestration failures; no frameworks, policy calls or physical execution."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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


def test_stable_admission_needs_thirteen_samples():
    ready, rows = admit([8 << 30]*13)
    assert ready and len(rows) == 13 and rows[-1]['elapsed_s'] == 60


def test_low_samples_reset_continuous_window():
    ready, rows = admit([8 << 30]*12+[7 << 30]+[8 << 30]*13)
    assert ready and rows[-1]['elapsed_s'] == 125
    assert rows[12]['stable_elapsed_s'] is None


def test_isolated_peaks_do_not_admit():
    ready, rows = admit(([9 << 30]+[7 << 30]*11)*5+[9 << 30])
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


def test_large_scheduler_gap_resets_but_later_thirteen_samples_can_pass():
    clock = Clock()
    rows = []
    delays = iter([61]+[5]*12)
    assert p.stable_admission42(lambda: 8 << 30, rows.append, clock=clock,
                                sleeper=lambda _: clock.sleep(next(delays)))
    assert rows[-1]['elapsed_s'] == 121
    assert rows[-1]['consecutive_qualified_samples'] == 13


def test_slow_final_receipt_cannot_admit_after_wait_deadline():
    clock = Clock()
    rows = []
    def record(row):
        rows.append(row)
        if len(rows) == 13:
            clock.sleep(250)
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
