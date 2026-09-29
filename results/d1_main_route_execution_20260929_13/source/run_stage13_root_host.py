"""One-shot 600 s outer host; track only fresh descendant process identities."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def processes() -> dict[int, tuple[int, int, str]]:
    rows = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            rows[int(path.name)] = (int(fields[1]), int(fields[19]), fields[0])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


def save(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--launcher', type=Path, required=True)
    parser.add_argument('--arm', choices=('headless', 'gui'), required=True)
    parser.add_argument('--receipt-directory', type=Path, required=True)
    args = parser.parse_args()
    plan, launcher = args.plan.resolve(strict=True), args.launcher.resolve(strict=True)
    plan_value = json.loads(plan.read_text())
    output = Path(plan_value['arms'][args.arm]['output_directory'])
    folder = args.receipt_directory.resolve()
    folder.mkdir(exist_ok=False)
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(launcher),
               '--plan', str(plan), '--arm', args.arm, '--run']
    save(folder / 'reservation.json', {
        'scope': 'outer host; physical reservation remains inside launcher',
        'arm': args.arm, 'command': command, 'overall_wall_limit_s': 600,
        'terminate_at_s': 595, 'kill_at_s': 599, 'retry_permitted': False,
        'plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
        'host_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    tracked: dict[int, int] = {}
    stop = False
    child = None
    failure = None
    started = time.monotonic()
    # Orphaned setsid descendants reparent here, even between two /proc samples.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot enable owned-child subreaper')
    environment = os.environ.copy()
    environment['GUI13_PREFLIGHT_STARTED_MONOTONIC'] = str(started)
    ready_seen = False

    def interrupted(_number, _frame):
        nonlocal stop
        stop = True

    old = {number: signal.signal(number, interrupted)
           for number in (signal.SIGTERM, signal.SIGINT)}

    def discover() -> dict[int, tuple[int, int, str]]:
        rows = processes()
        for pid, (parent, birth, _state) in rows.items():
            if parent == os.getpid():
                tracked.setdefault(pid, birth)
        if child is not None and child.pid in rows:
            tracked.setdefault(child.pid, rows[child.pid][1])
        changed = True
        while changed:
            changed = False
            for pid, (parent, birth, _state) in rows.items():
                if (pid not in tracked and parent in tracked and parent in rows
                        and rows[parent][1] == tracked[parent]):
                    tracked[pid] = birth
                    changed = True
        return {pid: row for pid, row in rows.items()
                if tracked.get(pid) == row[1] and row[2] != 'Z'}

    def send(number: int) -> None:
        # Individual PID plus /proc start time avoids signalling another job's group.
        for pid in discover():
            try:
                os.kill(pid, number)
            except ProcessLookupError:
                pass

    try:
        with (folder / 'host_stdout.log').open('xb') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=environment)
            while child.poll() is None:
                owned = discover()
                elapsed = time.monotonic() - started
                if not ready_seen and (output / 'preflight_ready.json').is_file():
                    try:
                        ready = json.loads((output / 'preflight_ready.json').read_text())
                    except json.JSONDecodeError:
                        ready = None
                    if ready is not None:
                        ready_ns = ready.get('monotonic_ns')
                        if (ready.get('pid') not in owned or type(ready_ns) is not int
                                or not started <= ready_ns / 1e9 <= min(started + 240, time.monotonic())
                                or ready.get('session_sha256') != hashlib.sha256(
                                    (output / 'session.json').read_bytes()).hexdigest()
                                or ready.get('source_count') != len(plan_value['inputs']) + 1
                                or ready.get('full_source_sha256_checked') is not True
                                or ready.get('engine_imported') is not False
                                or ready.get('model_loaded') is not False):
                            raise RuntimeError('outer host rejected readiness identity')
                        ready_seen = True
                if not ready_seen and elapsed >= 240:
                    failure = 'outer_preflight_wall_limit'
                    break
                if stop or elapsed >= 595:
                    failure = 'outer_interrupted' if stop else 'outer_wall_limit'
                    break
                time.sleep(0.2)
    except BaseException as error:  # noqa: BLE001 - close only this host's owned children
        failure = f'{type(error).__name__}: {error}'
    finally:
        if discover():
            send(signal.SIGTERM)
            deadline = min(started + 599, time.monotonic() + 4)
            while discover() and time.monotonic() < deadline:
                time.sleep(0.05)
            if discover():
                send(signal.SIGKILL)
        if child is not None:
            try:
                child.wait(timeout=max(0.01, min(1.0, started + 600 - time.monotonic())))
            except subprocess.TimeoutExpired:
                failure = failure or 'outer_child_not_reaped'
        survivors = discover()
        for pid in tracked:
            if child is None or pid != child.pid:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        receipt = {
            'arm': args.arm, 'command': command, 'failure': failure,
            'exit_code': None if child is None else child.returncode,
            'elapsed_wall_s': time.monotonic() - started,
            'tracked_process_birth_ticks': tracked,
            'surviving_owned_processes': survivors, 'no_orphans': not survivors,
            'linux_child_subreaper': True, 'preflight_ready_seen': ready_seen,
            'retry_permitted': False}
        save(folder / 'receipt.json', receipt)
        for number, handler in old.items():
            signal.signal(number, handler)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if (failure is None and ready_seen and receipt['exit_code'] == 0
                 and not survivors and receipt['elapsed_wall_s'] <= 600) else 1


if __name__ == '__main__':
    raise SystemExit(main())
