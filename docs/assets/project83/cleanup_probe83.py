"""Owned-process cleanup fixture; run only under the new83 supervisor.

The sole child sleeps30 seconds in a new session/process group. This fixture
never signals a process. It deliberately leaves cleanup to its supervisor.
Neither fixture mode proves cleanup: only the supervisor's later receipt can.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write_new(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if len(payload.encode()) > 16384:
        raise ValueError('fixture record exceeds16KiB')
    with path.open('x') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def process_record(pid):
    folder = Path('/proc') / str(pid)
    status = {}
    for line in (folder / 'status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('Pid', 'PPid'):
            status[key] = int(value.strip())
    groups = [line[3:] for line in (folder / 'cgroup').read_text().splitlines()
              if line.startswith('0::')]
    if len(groups) != 1 or status.get('Pid') != pid:
        raise ValueError('unambiguous live process identity required')
    return {'pid': pid, 'ppid': status['PPid'], 'pgid': os.getpgid(pid),
            'session_id': os.getsid(pid), 'unified_cgroup': groups[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('orphan', 'timeout'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    initial = {'schema': 'owned-cleanup-probe83-v1', 'status': 'INCOMPLETE',
               'mode': args.mode, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'new_model_calls': 0, 'new_physics_steps': 0, 'signals_sent_by_fixture': 0,
               'cleanup_verified_by_fixture': False}
    write_new(args.output / 'initial.json', initial)
    child = None
    try:
        parent = process_record(os.getpid())
        child = subprocess.Popen(
            [sys.executable, '-B', '-c', 'import time; time.sleep(30)'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True,
        )
        descendant = process_record(child.pid)
        if (descendant['ppid'] != parent['pid'] or descendant['pgid'] == parent['pgid']
                or descendant['pgid'] != child.pid or descendant['session_id'] != child.pid
                or descendant['unified_cgroup'] != parent['unified_cgroup']):
            raise ValueError('fixture child is not the intended owned independent process group')
        if child.poll() is not None:
            raise ValueError('fixture child exited before supervisor handoff')
        receipt = {**initial, 'status': 'READY_FOR_SUPERVISOR_CLEANUP',
                   'parent': parent, 'child': descendant, 'child_sleep_seconds': 30,
                   'recorded_monotonic_ns': time.monotonic_ns(),
                   'expected_parent_action': 'exit0_without_waiting' if args.mode == 'orphan'
                   else 'sleep30_until_supervisor_timeout',
                   'source_unchanged': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                   == initial['script_sha256']}
        if not receipt['source_unchanged']:
            raise ValueError('fixture source changed')
        write_new(args.output / 'spawned.json', receipt)
        print(json.dumps(receipt, allow_nan=False), flush=True)
        if args.mode == 'orphan':
            return 0
        # Default SIGTERM behavior is intentional: the supervisor must retain
        # its timeout/failure receipt and clean both independent process groups.
        time.sleep(30)
        write_new(args.output / 'unexpected_timeout_survival.json',
                  {**initial, 'failure': 'parent survived expected supervisor3s timeout'})
        return 2
    except Exception as exc:  # noqa: BLE001 - retain the failed fixture prefix for its supervisor
        write_new(args.output / 'failure.json',
                  {**initial, 'error': type(exc).__name__ + ': ' + str(exc),
                   'spawned_child_pid': None if child is None else child.pid,
                   'cleanup_owner': 'supervisor; fixture does not signal any process'})
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
