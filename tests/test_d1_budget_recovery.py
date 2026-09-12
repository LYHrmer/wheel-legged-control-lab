"""Recovery contracts with tiny artifacts; no training or physical simulation."""

import copy
import csv
import io
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import recover_d1_budget_study as recovery
from scripts import run_d1_budget_study as frozen


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def write_ledgers(study, records):
    (study / 'runs.jsonl').write_bytes(b''.join(
        (json.dumps(row, ensure_ascii=False) + '\n').encode() for row in records))
    with (study / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=frozen.LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(recovery.csv_record(row, frozen.LEDGER_FIELDS) for row in records)


@pytest.fixture
def interrupted(tmp_path, monkeypatch):
    repo, study, archive = tmp_path / 'repo', tmp_path / 'study', tmp_path / 'second_archive'
    prior = tmp_path / 'first_archive'
    repo.mkdir()
    study.mkdir()
    hashes = {}
    for name in (recovery.RUNNER, recovery.CHILD, 'extra.txt'):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + '\n')
        hashes[name] = recovery.digest(path)
    child_hashes = {recovery.CHILD: hashes[recovery.CHILD]}

    def child_artifacts(path, mode):
        path.mkdir(parents=True)
        (path / 'source.tar.gz').write_bytes(b'tiny archived bytes')
        write_json(path / 'source.json', {'sha256': child_hashes,
                   'archive_sha256': recovery.digest(path / 'source.tar.gz')})
        write_json(path / 'protocol.json', {'mode': mode})
        write_json(path / ('training.json' if mode == 'train' else 'evaluation.json'), {'ok': True})
        (path / 'episode.csv').write_bytes(b'also preserve complete child data\n')

    runs = []
    for sequence in range(1, 57):
        mode = 'train' if sequence <= 6 else 'evaluate'
        name = f'run{sequence:02d}'
        runs.append({'name': name, 'mode': mode, 'phase': 'train' if mode == 'train' else 'holdout',
                     'command': ['/usr/bin/python3', str(repo / recovery.CHILD), mode,
                                 '--output', str(study / name)]})
    protocol = {'kind': 'formal', 'runs': runs, 'source_sha256': hashes,
                'working_directory': str(repo), 'planned_commands': 56,
                'python_executable': '/usr/bin/python3', 'child_environment_overrides': {},
                'planned_evaluation_cases': 200}
    write_json(study / 'protocol.json', protocol)
    write_json(study / 'source.json', {'sha256': hashes, 'archived_extras': {
        recovery.RUNNER: 'orchestrator_source.py', 'extra.txt': 'source_extras/extra.txt'}})
    shutil.copyfile(repo / recovery.RUNNER, study / 'orchestrator_source.py')
    (study / 'source_extras').mkdir()
    shutil.copyfile(repo / 'extra.txt', study / 'source_extras/extra.txt')
    (study / 'logs').mkdir()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = []
    for sequence, item in enumerate(runs[:40], 1):
        row = {**item, 'sequence': sequence, 'returncode': 0,
               'started_utc': (start + timedelta(seconds=2 * sequence)).isoformat(),
               'ended_utc': (start + timedelta(seconds=2 * sequence + 1)).isoformat(),
               'elapsed_s': 1.0, 'log': f"logs/{item['name']}.log"}
        (study / row['log']).write_text('completed\n' if sequence < 40 else 'interrupted\n')
        child_artifacts(study / item['name'], item['mode'])
        if sequence < 40:
            row['child_artifact_sha256'] = frozen._child_artifacts(study / item['name'], item['mode'])
        else:
            row.update(returncode=None, error_type='KeyboardInterrupt', error='received signal 15')
        records.append(row)
    write_ledgers(study, records)
    prior.mkdir()
    for name in ('protocol.json', 'source.json', 'orchestrator_source.py'):
        shutil.copyfile(study / name, prior / name)
    shutil.copytree(study / 'source_extras', prior / 'source_extras')
    (prior / 'logs').mkdir()
    for row in records[:35]:
        shutil.copytree(study / row['name'], prior / row['name'])
        shutil.copyfile(study / row['log'], prior / row['log'])
    write_ledgers(prior, records[:35])
    (prior / 'run36').mkdir()
    (prior / 'run36/partial').write_text('first interruption')
    (prior / 'logs/run36.log').write_text('first interruption')
    snapshot = recovery.inventory(prior)
    copied = [name for name in snapshot if not name.startswith('run36/') and name != 'logs/run36.log']
    old_recovery = study / 'recovery'
    old_recovery.mkdir()
    (old_recovery / 'helper_source.py').write_bytes(b'old helper exact bytes\n')
    (old_recovery / 'host_idle_evidence').write_bytes(b'old host evidence\n')
    write_json(old_recovery / 'plan.json', {
        'schema': 'd1-interrupted-study-recovery-v1', 'study': str(study), 'archive': str(prior),
        'completed_commands': 35, 'frozen_source_sha256': hashes,
        'original_files': snapshot, 'copy_files': copied})
    write_json(old_recovery / 'provenance.json', {
        'archive': str(prior), 'reused_commands': 35,
        'helper_sha256': recovery.digest(old_recovery / 'helper_source.py'),
        'host_idle_evidence_sha256': recovery.digest(old_recovery / 'host_idle_evidence')})
    write_json(old_recovery / 'failure.json', {
        'status': 'failed', 'commands_completed': 39, 'planned_commands': 56,
        'error_type': 'KeyboardInterrupt', 'message': 'received signal 15'})
    write_json(study / 'failure.json', {'status': 'failed', 'recovery_failure': 'recovery/failure.json'})
    old_lock = tmp_path / '.study.recovery.lock'
    old_lock.write_bytes(b'old persistent lock\n')
    evidence = tmp_path / 'host_evidence'
    evidence.write_bytes(b'test host evidence\n')
    calls = []

    def build_runs(output, smoke=False):
        planned = copy.deepcopy(runs)
        for item in planned:
            item['command'][-1] = str(output / item['name'])
        return planned

    def execute(command, environment, log):
        calls.append(command)
        assert command[2] == 'evaluate'
        child_artifacts(Path(command[-1]), 'evaluate')
        return 0

    runner = SimpleNamespace(
        LEDGER_FIELDS=frozen.LEDGER_FIELDS, build_runs=build_runs,
        source_hashes=lambda: {name: recovery.digest(repo / name) for name in hashes},
        _child_artifacts=frozen._child_artifacts, _environment=lambda: {}, _execute=execute,
        _utc=lambda: datetime.now(timezone.utc).isoformat(), _sigterm=frozen._sigterm,
        write_json=frozen.write_json)
    monkeypatch.setattr(recovery, 'SOURCE_COUNT', 3)
    monkeypatch.setattr(recovery, 'frozen_runner', lambda *args: runner)
    monkeypatch.setattr(recovery, 'assert_host_idle', lambda: None)
    return SimpleNamespace(repo=repo, study=study, archive=archive, prior=prior,
                           records=records, runner=runner, evidence=evidence, calls=calls,
                           old_lock=old_lock, hashes=hashes)


def test_csv_prefix_opus_core_unicode_and_multiline():
    raw = 'id,note\r\n1,"甲\r\n乙"\r\n2,"换行\n尾部"\r\n3,failed\r\n'.encode()
    assert recovery.csv_prefix_bytes(raw, 2) == raw[:raw.index(b'3,failed')]
    assert recovery.csv_prefix_bytes(raw, 0) == b'id,note\r\n'
    for invalid in (True, 1.5, -1):
        with pytest.raises(ValueError):
            recovery.csv_prefix_bytes(raw, invalid)


def test_full_ledger_prefix_keeps_multiline_csv_bytes(interrupted):
    f = interrupted
    f.records[38]['log'] = 'logs/甲\r\n乙\n尾.log'
    write_ledgers(f.study, f.records)
    raw = (f.study / 'runs.csv').read_bytes()
    good, failed, jsonl, csv_bytes = recovery.read_ledger_prefix(f.study, 39, frozen.LEDGER_FIELDS)
    assert len(good) == 39 and failed['sequence'] == 40
    assert len(jsonl.splitlines()) == 39
    assert csv_bytes == raw[:raw.index(b'40,run40,')]
    assert len(list(csv.DictReader(io.StringIO(csv_bytes.decode(), newline='')))) == 39


@pytest.mark.parametrize('change', ['tail_success', 'tail_sequence', 'prefix_failure', 'bool_sequence', 'csv_tail', 'extra_csv_cell', 'partial_jsonl'])
def test_rejects_invalid_ledger_including_failed_tail(interrupted, change):
    f = interrupted
    if change == 'tail_success':
        f.records[-1]['returncode'] = 0
    elif change == 'tail_sequence':
        f.records[-1]['sequence'] = 41
    elif change == 'prefix_failure':
        f.records[-2]['returncode'] = 1
    elif change == 'bool_sequence':
        f.records[0]['sequence'] = True
    write_ledgers(f.study, f.records)
    if change == 'csv_tail':
        path = f.study / 'runs.csv'
        path.write_bytes(path.read_bytes().replace(b'received signal 15', b'received signal 9'))
    elif change == 'extra_csv_cell':
        path = f.study / 'runs.csv'
        path.write_bytes(path.read_bytes()[:-2] + b',extra\r\n')
    elif change == 'partial_jsonl':
        path = f.study / 'runs.jsonl'
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        recovery.read_ledger_prefix(f.study, 39, frozen.LEDGER_FIELDS)


def test_validate_is_read_only_and_plans_40_through_56(interrupted):
    f = interrupted
    before = recovery.inventory(f.study)
    _, protocol, records, plan = recovery.validate(f.repo, f.study, f.archive, 39)
    assert len(records) == 39 and len(protocol['runs']) == 56
    assert plan['remaining_sequences'] == list(range(40, 57))
    assert not any(name.startswith(('run40/', 'recovery/')) or name == 'failure.json' for name in plan['copy_files'])
    assert before == recovery.inventory(f.study)
    assert not f.archive.exists()


@pytest.mark.parametrize('change', ['source', 'archive', 'copied', 'helper', 'old_lock', 'occupied_archive', 'wrong_count'])
def test_validate_rejects_source_chain_changes_and_clobber(interrupted, change):
    f = interrupted
    if change == 'source':
        (f.repo / recovery.CHILD).write_bytes(b'changed')
    elif change == 'archive':
        (f.prior / 'run01/episode.csv').write_bytes(b'changed')
    elif change == 'copied':
        (f.study / 'run01/episode.csv').write_bytes(b'changed')
    elif change == 'helper':
        (f.study / 'recovery/helper_source.py').write_bytes(b'changed')
    elif change == 'old_lock':
        f.old_lock.unlink()
    elif change == 'occupied_archive':
        f.archive.mkdir()
    with pytest.raises(ValueError):
        recovery.validate(f.repo, f.study, f.archive, 38 if change == 'wrong_count' else 39)


def test_execute_preserves_archive_and_exact_39_prefix_without_retraining(interrupted):
    f = interrupted
    before = recovery.inventory(f.study)
    jsonl_prefix = b''.join((f.study / 'runs.jsonl').read_bytes().splitlines(keepends=True)[:39])
    csv_prefix = recovery.csv_prefix_bytes((f.study / 'runs.csv').read_bytes(), 39)
    result = recovery.execute(f.repo, f.study, f.archive, f.evidence, 39)
    assert result['commands_completed'] == 56 and result['reused_commands'] == 39
    assert [Path(command[-1]).name for command in f.calls] == [f'run{i:02d}' for i in range(40, 57)]
    assert (f.study / 'runs.jsonl').read_bytes().startswith(jsonl_prefix)
    assert (f.study / 'runs.csv').read_bytes().startswith(csv_prefix)
    assert len((f.study / 'runs.jsonl').read_bytes().splitlines()) == 56
    assert recovery.inventory(f.archive) == before
    assert (f.archive / 'failure.json').is_file() and (f.archive / 'recovery/failure.json').is_file()
    assert not (f.study / 'failure.json').exists()
    assert f.old_lock.read_bytes() == b'old persistent lock\n'
    assert (f.study.parent / '.study.recovery-second_archive.lock').is_file()
    assert recovery.read_json(f.study / 'source_consistency.json')['sha256'] == f.hashes


def test_child_failure_is_retained_without_retry_or_summary(interrupted):
    f = interrupted

    def fail(command, environment, log):
        f.calls.append(command)
        return 7

    f.runner._execute = fail
    with pytest.raises(ValueError, match='child exited 7'):
        recovery.execute(f.repo, f.study, f.archive, f.evidence, 39)
    assert len(f.calls) == 1
    assert not (f.study / 'summary.json').exists()
    assert recovery.read_json(f.study / 'recovery/failure.json')['commands_completed'] == 39
    tail = json.loads((f.study / 'runs.jsonl').read_bytes().splitlines()[-1])
    assert tail['sequence'] == 40 and tail['returncode'] == 7


def test_source_change_before_first_resumed_child_aborts(interrupted):
    f = interrupted

    def environment():
        (f.repo / recovery.CHILD).write_bytes(b'changed after copying')
        return {}

    f.runner._environment = environment
    with pytest.raises(ValueError, match='source changed before resumed command'):
        recovery.execute(f.repo, f.study, f.archive, f.evidence, 39)
    assert not f.calls and not (f.study / 'summary.json').exists()
    assert recovery.read_json(f.study / 'recovery/failure.json')['commands_completed'] == 39


def test_rename_does_not_replace_existing_archive(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    target.mkdir()
    (target / 'sentinel').write_bytes(b'keep')
    with pytest.raises(OSError):
        recovery.rename_noreplace(source, target)
    assert source.exists() and (target / 'sentinel').read_bytes() == b'keep'


def test_existing_new_lock_prevents_any_archive_mutation(interrupted):
    f = interrupted
    lock = f.study.parent / '.study.recovery-second_archive.lock'
    lock.write_bytes(b'previous attempt\n')
    before = recovery.inventory(f.study)
    with pytest.raises(FileExistsError):
        recovery.execute(f.repo, f.study, f.archive, f.evidence, 39)
    assert not f.archive.exists() and recovery.inventory(f.study) == before
    assert lock.read_bytes() == b'previous attempt\n'


@pytest.mark.parametrize('other_command', [
    b'python3\0/other/recover_d1_budget_study.py\0',
    b'/usr/bin/python3.10\0-B\0/other/recover_d1_budget_study.py\0',
    b'python3\0-m\0scripts.recover_d1_budget_study\0',
])
def test_host_scan_skips_self_and_wrappers_but_blocks_other_helper(tmp_path, monkeypatch, other_command):
    own = tmp_path / '123/cmdline'
    own.parent.mkdir()
    own.write_bytes(b'python3\0/some/recover_d1_budget_study.py\0')
    for pid, command in [('450', b'rtk\0proxy\0python3\0/some/recover_d1_budget_study.py\0'),
                         ('451', b'systemd-run\0--user\0python3\0/some/recover_d1_budget_study.py\0')]:
        wrapper = tmp_path / pid / 'cmdline'
        wrapper.parent.mkdir()
        wrapper.write_bytes(command)

    class HostPath(type(Path())):
        def read_text(self, *args, **kwargs):
            if str(self) == '/proc/1/comm':
                return 'systemd\n'
            return super().read_text(*args, **kwargs)

        def glob(self, pattern):
            if str(self) == '/proc':
                return HostPath(tmp_path).glob(pattern.replace('[0-9]*', '*'))
            return super().glob(pattern)

    monkeypatch.setattr(recovery, 'Path', HostPath)
    monkeypatch.setattr(recovery.os, 'getpid', lambda: 123)
    recovery.assert_host_idle()
    other = tmp_path / '456/cmdline'
    other.parent.mkdir()
    other.write_bytes(other_command)
    with pytest.raises(ValueError, match='experiment process still active: 456'):
        recovery.assert_host_idle()
