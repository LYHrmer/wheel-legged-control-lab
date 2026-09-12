"""One explicit recovery of the frozen formal study; report-only by default.

Preserve the entire interrupted directory by a no-clobber rename, byte-copy the
explicitly completed runs back, then rerun the failed evaluation and remainder.
The original runner and frozen 77 source files are never edited. No retries.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
TOTAL, SOURCE_COUNT = 56, 77
RUNNER = 'scripts/run_d1_budget_study.py'
CHILD = 'scripts/run_d1_locomotion_experiment.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def regular_path(path):
    path = Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            f'path traversal/symlink rejected: {path}')
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def inventory(root):
    require(root.is_dir(), f'missing input directory: {root}')
    result = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), f'symlink in preserved study: {path}')
        if not path.is_dir():
            require(path.is_file(), f'non-regular study artifact: {path}')
            result[path.relative_to(root).as_posix()] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    return result


def frozen_runner(repo, study, hashes):
    require(digest(repo / RUNNER) == hashes[RUNNER], 'current runner differs from frozen SHA')
    require(digest(study / 'orchestrator_source.py') == hashes[RUNNER], 'archived runner differs')
    spec = importlib.util.spec_from_file_location('d1_frozen_recovery_runner', study / 'orchestrator_source.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = repo
    return runner


def csv_record(record, fields):
    return {**{key: record.get(key) for key in fields},
            'command_json': json.dumps(record['command']),
            'child_artifact_sha256_json': json.dumps(record.get('child_artifact_sha256'))}


def csv_prefix_bytes(raw, completed_rows):
    """Byte-exact logical CSV prefix; core contributed by Claude Opus.

    Opus session 1b05f559-f405-425b-ae77-3ab321a46b5d supplied the physical-line
    to byte-offset mapping. Full ledger validation is separate below.
    """
    require(type(completed_rows) is int and completed_rows >= 0, 'invalid CSV row count')
    lines = re.findall(r'[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+', raw.decode('utf-8'))
    require(lines, 'missing CSV header')
    ends, total = [], 0
    for line in lines:
        total += len(line.encode('utf-8'))
        ends.append(total)
    reader = csv.reader(iter(lines), strict=True)
    try:
        header = next(reader)
        require(header and any(field.strip() for field in header), 'missing CSV header')
        require(len(header) == len(set(header)), 'duplicate CSV header')
        end_line, found = reader.line_num, 0
        while found < completed_rows:
            record = next(reader)
            if record:
                found += 1
                end_line = reader.line_num
    except (csv.Error, StopIteration) as exc:
        raise ValueError('malformed/incomplete CSV prefix') from exc
    return raw[:ends[end_line - 1]]


def read_ledger_prefix(study, completed, fields):
    """Validate every success and the failed tail, then retain original bytes."""
    require(type(completed) is int and completed > 0, 'invalid explicit completed count')
    raw_jsonl = (study / 'runs.jsonl').read_bytes()
    lines = raw_jsonl.splitlines(keepends=True)
    require(lines and all(line.endswith(b'\n') for line in lines), 'partial JSONL line')
    records = [json.loads(line, object_pairs_hook=unique_object) for line in lines]
    require(len(records) == completed + 1, 'expected successful prefix plus exactly one failed tail')
    for sequence, record in enumerate(records, 1):
        require(type(record['sequence']) is int and record['sequence'] == sequence, 'nonconsecutive ledger')
        if sequence <= completed:
            require(type(record['returncode']) is int and record['returncode'] == 0
                    and not record.get('error') and not record.get('error_type'), 'invalid successful prefix')
        else:
            require(record['returncode'] is None or (type(record['returncode']) is int and record['returncode'] != 0),
                    'tail is not a failed command')
            require(record.get('error') and record.get('error_type'), 'failed tail lacks error evidence')
    raw_csv = (study / 'runs.csv').read_bytes()
    require(raw_csv.endswith((b'\n', b'\r')), 'partial CSV final line')
    try:
        reader = csv.DictReader(io.StringIO(raw_csv.decode('utf-8'), newline=''), strict=True)
        require(tuple(reader.fieldnames or ()) == tuple(fields), 'CSV header differs')
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), 'duplicate CSV header')
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError('malformed CSV ledger') from exc
    require(len(rows) == len(records), 'CSV/JSONL count differs')
    for record, row in zip(records, rows):
        require(None not in row and None not in row.values(), 'malformed CSV record width')
        expected = csv_record(record, fields)
        for key, value in row.items():
            if key in ('command_json', 'child_artifact_sha256_json'):
                require(json.loads(value, object_pairs_hook=unique_object)
                        == json.loads(expected[key], object_pairs_hook=unique_object), f'CSV {key} differs')
            else:
                require(value == ('' if expected[key] is None else str(expected[key])), f'CSV {key} differs')
    return records[:-1], records[-1], b''.join(lines[:completed]), csv_prefix_bytes(raw_csv, completed)


def validate_previous_recovery(study, completed, hashes):
    """Accept the recorded 35-to-39 recovery interrupted by SIGTERM only."""
    require(completed == 39, 'this reviewed recovery requires exactly 39 completed commands')
    recovery = study / 'recovery'
    require({p.name for p in recovery.iterdir()} == {
        'plan.json', 'provenance.json', 'helper_source.py', 'host_idle_evidence', 'failure.json'},
        'unexpected previous recovery artifacts')
    plan, provenance, failure = (read_json(recovery / name) for name in
                                 ('plan.json', 'provenance.json', 'failure.json'))
    require(read_json(study / 'failure.json') == {'status': 'failed', 'recovery_failure': 'recovery/failure.json'},
            'previous root failure marker differs')
    require(plan['schema'] == 'd1-interrupted-study-recovery-v1'
            and plan['study'] == str(study) and plan['completed_commands'] == 35
            and provenance['reused_commands'] == 35 and plan['frozen_source_sha256'] == hashes,
            'previous recovery plan differs')
    require(failure['status'] == 'failed' and failure['commands_completed'] == completed
            and failure['planned_commands'] == TOTAL and failure['error_type'] == 'KeyboardInterrupt'
            and failure['message'] == 'received signal 15', 'previous failure is not the reviewed SIGTERM')
    archive = regular_path(plan['archive'])
    require(archive.parent == study.parent and archive != study
            and provenance['archive'] == str(archive), 'previous archive path differs')
    require(digest(recovery / 'helper_source.py') == provenance['helper_sha256'], 'previous helper SHA differs')
    require(digest(recovery / 'host_idle_evidence') == provenance['host_idle_evidence_sha256'],
            'previous host evidence SHA differs')
    require(inventory(archive) == plan['original_files'], 'previous archive inventory differs')
    for name in ('runs.jsonl', 'runs.csv'):
        old = (archive / name).read_bytes()
        current = (study / name).read_bytes()
        prefix = b''.join(current.splitlines(keepends=True)[:35]) if name.endswith('jsonl') else csv_prefix_bytes(current, 35)
        require(old == prefix, f'previous {name} prefix differs')
    for name in plan['copy_files']:
        if name in ('runs.jsonl', 'runs.csv'):
            continue
        original = regular_path(archive / name)
        current = regular_path(study / name)
        require(original.is_relative_to(archive) and current.is_relative_to(study), 'previous copy path escapes study')
        expected = plan['original_files'][name]
        require(current.stat().st_size == expected['bytes'] and digest(current) == expected['sha256'],
                f'previous copied artifact changed: {name}')
    lock = regular_path(study.parent / f'.{study.name}.recovery.lock')
    require(lock.is_file(), 'original persistent recovery lock missing')
    return {'archive': str(archive), 'archived_recovery_directory': 'recovery',
            'original_archive_files_verified': len(plan['original_files']),
            'reused_commands_in_previous_recovery': 35,
            'metadata_sha256': {p.name: digest(p) for p in sorted(recovery.iterdir())},
            'original_lock': str(lock), 'original_lock_sha256': digest(lock)}


def validate(repo, study, archive, completed):
    require(type(completed) is int and 6 <= completed < TOTAL, "invalid explicit completed count")
    repo, study, archive = map(regular_path, (repo, study, archive))
    require(study.parent == archive.parent and study != archive, 'archive must be a distinct sibling')
    require(not os.path.lexists(archive), 'archive exists; never overwrite')
    protocol = read_json(study / 'protocol.json')
    source = read_json(study / 'source.json')
    hashes = protocol['source_sha256']
    require(source['sha256'] == hashes and len(hashes) == SOURCE_COUNT, 'frozen source set mismatch')
    runner = frozen_runner(repo, study, hashes)
    require(runner.source_hashes() == hashes, 'current frozen source inventory differs')
    require(protocol['kind'] == 'formal' and len(protocol['runs']) == TOTAL, 'expected fixed formal 56-command protocol')
    require(protocol['working_directory'] == str(repo), 'recorded repository differs')
    require(protocol['planned_commands'] == TOTAL, 'planned command count differs')
    # Rebuild the frozen argv on an unused sibling, then translate only that
    # directory prefix back. This avoids disabling the runner's no-clobber guard.
    expected_runs = runner.build_runs(archive, smoke=False)
    for item in expected_runs:
        item['command'] = [str(study) + part[len(str(archive)):]
                           if part.startswith(str(archive) + '/') else part
                           for part in item['command']]
    require(protocol['runs'] == expected_runs, 'planned commands differ from frozen runner')
    for name, relative in source['archived_extras'].items():
        require(digest(regular_path(study / relative)) == hashes[name], 'parent archived extra SHA differs')
    for forbidden in ('summary.json', 'source_consistency.json'):
        require(not os.path.lexists(study / forbidden), f'already finalized: {forbidden}')
    chain = validate_previous_recovery(study, completed, hashes)
    records, failed, jsonl_prefix, csv_prefix = read_ledger_prefix(study, completed, runner.LEDGER_FIELDS)
    require(failed['returncode'] is None and failed['error_type'] == 'KeyboardInterrupt'
            and failed['error'] == 'received signal 15', 'failed tail is not the reviewed SIGTERM')
    with (study / 'runs.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))[:completed]
    names = set()
    common_child_source = None
    previous_end = None
    for sequence, item in enumerate(protocol['runs'], 1):
        name, command = item['name'], item['command']
        require(isinstance(name, str) and Path(name).name == name and name not in names, 'invalid/duplicate run name')
        names.add(name)
        require(command[:3] == [protocol['python_executable'], str(repo / CHILD), item['mode']], 'child executable/mode differs')
        require(command.count('--output') == 1, 'ambiguous output argument')
        require(command[command.index('--output') + 1] == str(study / name), 'recorded child path differs')
        for flag in ('--policy', '--metadata'):
            if flag in command:
                require(command.count(flag) == 1, 'ambiguous checkpoint argument')
                checkpoint = regular_path(command[command.index(flag) + 1])
                require(checkpoint.is_relative_to(study) and checkpoint.is_file(), 'missing/outside checkpoint')
        if sequence > completed:
            require(item['mode'] == 'evaluate' and item['phase'] == 'holdout', 'remaining command is not holdout evaluate')
            continue
        record, csv_row = records[sequence - 1], csv_rows[sequence - 1]
        require(all(record.get(key) == value for key, value in item.items()), 'ledger differs from planned command')
        require(type(record['sequence']) is int and record['sequence'] == sequence, 'nonconsecutive ledger')
        require(type(record['returncode']) is int and record['returncode'] == 0, 'failed/nonzero old command')
        require(not record.get('error') and not record.get('error_type'), 'old command reports an error')
        require(record['log'] == f'logs/{name}.log' and (study / record['log']).is_file(), 'old log missing')
        start, end = (datetime.fromisoformat(record[k]) for k in ('started_utc', 'ended_utc'))
        require(start.utcoffset() is not None and end.utcoffset() is not None and end >= start,
                'invalid old timestamps')
        require(previous_end is None or start >= previous_end, 'reordered/overlapping old commands')
        previous_end = end
        elapsed = record['elapsed_s']
        require(type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed >= 0, 'invalid old elapsed')
        require(runner._child_artifacts(study / name, item['mode']) == record['child_artifact_sha256'], 'old child artifact SHA differs')
        child_source = read_json(study / name / 'source.json')['sha256']
        require(child_source and all(hashes.get(key) == value for key, value in child_source.items()),
                'old child source differs from frozen source')
        if common_child_source is None:
            common_child_source = child_source
        require(child_source == common_child_source, 'old child source inventories differ')
        expected_csv = csv_record(record, runner.LEDGER_FIELDS)
        require(None not in csv_row and None not in csv_row.values(), 'malformed old CSV')
        for key, value in csv_row.items():
            if key in ('command_json', 'child_artifact_sha256_json'):
                require(json.loads(value) == json.loads(expected_csv[key]), f'CSV {key} differs')
            else:
                require(value == ('' if expected_csv[key] is None else str(expected_csv[key])), f'CSV {key} differs')
    require(sum(row['mode'] == 'train' for row in records) == 6, 'six completed training trajectories required')
    require(set(common_child_source) | set(source['archived_extras']) == set(hashes),
            'child archives plus parent extras do not cover all frozen source')
    pending_item = protocol['runs'][completed]
    pending = pending_item['name']
    require(all(failed.get(key) == value for key, value in pending_item.items()), 'failed tail differs from planned command')
    require(failed['log'] == f'logs/{pending}.log', 'failed tail log differs')
    start, end = (datetime.fromisoformat(failed[k]) for k in ('started_utc', 'ended_utc'))
    require(start.utcoffset() is not None and end.utcoffset() is not None
            and end >= start >= previous_end, 'invalid failed tail timestamps')
    require(type(failed['elapsed_s']) in (float, int) and math.isfinite(failed['elapsed_s'])
            and failed['elapsed_s'] >= 0, 'invalid failed tail elapsed')
    require((study / pending).is_dir(), 'expected interrupted command directory missing')
    require({p.name for p in (study / 'logs').iterdir()} == {f"{r['name']}.log" for r in records} | {f'{pending}.log'},
            'unexpected/missing original logs; human review required')
    for item in protocol['runs'][completed + 1:]:
        require(not os.path.lexists(study / item['name']), 'later unledgered child exists')
    allowed = {'protocol.json', 'source.json', 'orchestrator_source.py', 'runs.jsonl', 'runs.csv', 'source_extras', 'logs',
               'failure.json', 'recovery', pending, *(row['name'] for row in records)}
    require({p.name for p in study.iterdir()} == allowed, 'unknown/missing original root artifact; preserve and review')
    snapshot = inventory(study)
    copied_prefixes = {row['name'] for row in records} | {'source_extras'}
    copied_logs = {row['log'] for row in records}
    copied = [name for name in snapshot if name in ('protocol.json', 'source.json', 'orchestrator_source.py')
              or name.split('/')[0] in copied_prefixes or name in copied_logs]
    needed_bytes = sum(snapshot[name]['bytes'] for name in copied) + len(jsonl_prefix) + len(csv_prefix)
    require(shutil.disk_usage(study.parent).free > needed_bytes + 128 * 1024 * 1024, 'insufficient free space for exact copy')
    plan = {'schema': 'd1-interrupted-study-recovery-v2', 'study': str(study), 'archive': str(archive),
            'completed_commands': completed, 'previous_recovery': chain,
            'ledger_prefixes': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                                for name, data in [('runs.jsonl', jsonl_prefix), ('runs.csv', csv_prefix)]}, 'remaining_sequences': list(range(completed + 1, TOTAL + 1)),
            'remaining_names': [item['name'] for item in protocol['runs'][completed:]],
            'interrupted_run_preserved_only_in_archive': pending, 'copy_files': copied,
            'copied_bytes': needed_bytes, 'original_files': snapshot, 'frozen_source_sha256': hashes,
            'original_root_files_sha256': {name: data['sha256'] for name, data in snapshot.items() if '/' not in name},
            'original_command_elapsed_s_sum': sum(r['elapsed_s'] for r in records),
            'last_original_ended_utc': records[-1]['ended_utc'], 'original_first_started_utc': records[0]['started_utc']}
    return runner, protocol, records, plan


def assert_host_idle():
    # A private PID namespace can hide the very processes this gate must check.
    require(Path('/proc/1/comm').read_text().strip() in ('systemd', 'init'),
            'execute requires host /proc visibility; run outside sandbox after host process review')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if path.parent.name == str(os.getpid()):
            continue
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        args = [os.fsdecode(arg) for arg in args if arg]
        monitored = (Path(RUNNER).name, Path(CHILD).name,
                     'recover_d1_budget_study.py', 'helper_source.py')
        executable = Path(args[0]).name if args else ''
        # rtk/systemd-run may carry the whole launch argv while the service
        # starts. They are not experiment interpreters and must not block it.
        if not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', executable) and executable not in monitored:
            continue
        modules = {f'scripts.{Path(name).stem}' for name in monitored}
        require(not any(Path(arg).name in monitored or arg in modules for arg in args),
                f'experiment process still active: {path.parent.name}')


def rename_noreplace(source, target):
    function = ctypes.CDLL(None, use_errno=True).renameat2
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def append_record(runner, study, record):
    with (study / 'runs.jsonl').open('a') as ledger, (study / 'runs.csv').open('a', newline='') as stream:
        ledger.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        ledger.flush()
        os.fsync(ledger.fileno())
        csv.DictWriter(stream, fieldnames=runner.LEDGER_FIELDS).writerow(csv_record(record, runner.LEDGER_FIELDS))
        stream.flush()
        os.fsync(stream.fileno())


def execute(repo, study, archive, evidence_path, completed):
    require(evidence_path is not None, '--host-idle-evidence is required for execute')
    evidence_path = regular_path(evidence_path)
    evidence = evidence_path.read_bytes()
    assert_host_idle()
    runner, protocol, records, plan = validate(repo, study, archive, completed)
    study, archive = Path(plan['study']), Path(plan['archive'])
    lock = regular_path(study.parent / f'.{study.name}.recovery-{archive.name}.lock')
    with lock.open('x') as handle:
        handle.write(f'{os.getpid()}\n')
    # A persistent lock survives crashes; no automatic recovery of a recovery.
    started_utc, tick = runner._utc(), time.perf_counter()
    reused = completed
    hashes = plan['frozen_source_sha256']
    assert_host_idle()
    require(runner.source_hashes() == hashes, 'source changed before archive')
    require(inventory(study) == plan['original_files'], 'original changed after preflight')
    require(digest(Path(plan['previous_recovery']['original_lock']))
            == plan['previous_recovery']['original_lock_sha256'], 'original lock changed')
    rename_noreplace(study, archive)
    study.mkdir(exist_ok=False)
    recovery = study / 'recovery'
    recovery.mkdir()
    runner.write_json(recovery / 'plan.json', plan)
    (recovery / 'helper_source.py').write_bytes(Path(__file__).read_bytes())
    (recovery / 'host_idle_evidence').write_bytes(evidence)
    previous_sigterm = signal.signal(signal.SIGTERM, runner._sigterm)
    try:
        require(inventory(archive) == plan['original_files'], 'archive bytes differ from original')
        for name in plan['copy_files']:
            target = study / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with (archive / name).open('rb') as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
            require(digest(target) == plan['original_files'][name]['sha256'], f'copy SHA differs: {name}')
        _, _, jsonl_prefix, csv_prefix = read_ledger_prefix(archive, reused, runner.LEDGER_FIELDS)
        for name, data in [('runs.jsonl', jsonl_prefix), ('runs.csv', csv_prefix)]:
            require(hashlib.sha256(data).hexdigest() == plan['ledger_prefixes'][name]['sha256'], 'ledger prefix changed')
            with (study / name).open('xb') as stream:
                stream.write(data)
        (study / 'logs').mkdir(exist_ok=True)
        runner.write_json(recovery / 'provenance.json', {
            'started_utc': started_utc, 'archive': str(archive), 'original_protocol_copied_verbatim': True,
            'original_ledger_prefix_copied_verbatim': True, 'reused_commands': reused, 'previous_recovery': plan['previous_recovery'],
            'scope': f'rerun failed command{reused + 1} and continue through{TOTAL}; all evaluate; no training or selection',
            'helper_sha256': digest(recovery / 'helper_source.py'), 'host_idle_evidence_sha256': digest(recovery / 'host_idle_evidence'),
            'copy_verified_files': len(plan['copy_files']), 'archive_verified_files': len(plan['original_files']),
            'source_checks': 'original freeze matched preflight, before/after each resumed command and final gate',
            'unobserved_gap': 'no claim of continuous monitoring while original process was absent'})
        environment = runner._environment()
        require({key: environment.get(key) for key in protocol['child_environment_overrides']}
                == protocol['child_environment_overrides'], 'child environment overrides differ')
        for sequence, item in enumerate(protocol['runs'][reused:], reused + 1):
            require(runner.source_hashes() == hashes, 'source changed before resumed command')
            require(not os.path.lexists(study / item['name']), 'resume child directory exists; refusing retry')
            record = {**item, 'sequence': sequence, 'started_utc': runner._utc(), 'log': f"logs/{item['name']}.log"}
            require(datetime.fromisoformat(record['started_utc']) >= datetime.fromisoformat(records[-1]['ended_utc']), 'clock moved backwards')
            command_tick = time.perf_counter()
            print(f'[{sequence}/{TOTAL}] recovery evaluate {item["name"]}', flush=True)
            try:
                with (study / record['log']).open('x') as stream:
                    record['returncode'] = runner._execute(item['command'], environment, stream)
                require(record['returncode'] == 0, f'child exited {record["returncode"]}; no retry')
                record['child_artifact_sha256'] = runner._child_artifacts(study / item['name'], 'evaluate')
                require(runner.source_hashes() == hashes, 'source changed during resumed command')
            except BaseException as exc:
                record.setdefault('returncode', None)
                record.update(error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                record.update(ended_utc=runner._utc(), elapsed_s=time.perf_counter() - command_tick)
                append_record(runner, study, record)
            records.append(record)
            completed = sequence
        require(runner.source_hashes() == hashes, 'source changed at final gate')
        require(inventory(archive) == plan['original_files'], 'preserved archive changed')
        require(digest(Path(plan['previous_recovery']['original_lock']))
                == plan['previous_recovery']['original_lock_sha256'], 'original lock changed')
        runner.write_json(study / 'source_consistency.json', {'unchanged': True, 'sha256': hashes,
            'scope': 'frozen originals plus recovery preflight/every command/final checks; interruption gap not monitored',
            'recovery_provenance': 'recovery/provenance.json'})
        elapsed = time.perf_counter() - tick
        summary = {'status': 'commands_completed', 'kind': protocol['kind'], 'commands_completed': completed,
                   'planned_commands': TOTAL, 'planned_evaluation_cases': protocol['planned_evaluation_cases'],
                   'elapsed_s': sum(r['elapsed_s'] for r in records), 'elapsed_s_definition': 'sum of all56 recorded child command elapsed durations; excludes interruption and copying',
                   'recovery_elapsed_s': elapsed, 'recovery_started_utc': started_utc, 'recovery_ended_utc': runner._utc(),
                   'observed_gap_s': (datetime.fromisoformat(started_utc) - datetime.fromisoformat(plan['last_original_ended_utc'])).total_seconds(),
                   'quality_claim': 'none; independent full analyzer still required', 'reused_commands': reused}
        runner.write_json(study / 'summary.json', summary)
        return summary
    except BaseException as exc:
        runner.write_json(recovery / 'failure.json', {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc),
            'commands_completed': completed, 'planned_commands': TOTAL, 'elapsed_s': time.perf_counter() - tick,
            'note': 'archive and partial recovered output retained; no retry or completion claim'})
        runner.write_json(study / 'failure.json', {'status': 'failed', 'recovery_failure': 'recovery/failure.json'})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--completed', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--host-idle-evidence', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.execute:
            result = execute(args.repo, args.study, args.archive, args.host_idle_evidence, args.completed)
        else:
            _, _, _, plan = validate(args.repo, args.study, args.archive, args.completed)
            result = {key: value for key, value in plan.items() if key not in ('original_files', 'copy_files', 'frozen_source_sha256')}
            result.update(status='report_only_no_writes', preserved_file_count=len(plan['original_files']), copied_file_count=len(plan['copy_files']))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
