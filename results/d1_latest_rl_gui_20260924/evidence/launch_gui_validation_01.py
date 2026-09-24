"""Root's finite 07 acceptance launcher; one reservation per distinct profile."""
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/lyh/wheel-legged-control-lab')
WORK = Path(__file__).resolve().parent
GO = WORK / 'astra_gui_review_inputs_07.json'
PURE = WORK / 'pure_test_round_2_receipt.json'


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def cleanup_owned_worker(output):
    """Bound cleanup to the exact worker group and session recorded by this run."""
    path = output / 'child_pid.json'
    report = {'child_receipt_present': path.is_file(), 'signals_sent': [],
              'owned_group_alive_after_cleanup': False}
    if not path.is_file():
        return report
    child = json.loads(path.read_text())
    session = json.loads((output / 'session.json').read_text())
    if child['argv'] != session['worker_argv']:
        raise RuntimeError('cleanup refuses a worker argv outside the reserved session')
    group = child['pid']
    if type(group) is not int or group <= 1:
        raise RuntimeError('invalid owned worker group')

    def members():
        found = []
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit():
                continue
            try:
                status = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(status[2]) != group or status[0] == 'Z':
                    continue
                argv = (folder / 'cmdline').read_bytes().split(b'\0')
                if (str(ROOT / 'scripts/run_d1_latest_rl.py').encode() not in argv
                        or str(output / 'session.json').encode() not in argv):
                    raise RuntimeError('cleanup refuses a foreign process in the reused group')
                found.append(int(folder.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
        return found

    report['live_owned_pids_before'] = members()
    if report['live_owned_pids_before']:
        for number, deadline_s in ((signal.SIGTERM, 10), (signal.SIGKILL, 3)):
            if not members():
                break
            os.killpg(group, number)
            report['signals_sent'].append(int(number))
            deadline = time.monotonic() + deadline_s
            while time.monotonic() < deadline and members():
                time.sleep(.05)
    report['owned_group_alive_after_cleanup'] = bool(members())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('gui_policy_box', 'headless_zero_plane'), required=True)
    args = parser.parse_args()
    go = json.loads(GO.read_text())
    assert go['decision'] == 'GO'
    assert go['pure_test_receipt_sha256'] == hashlib.sha256(PURE.read_bytes()).hexdigest()
    for name, expected in go['inputs'].items():
        path = Path(name)
        assert path.stat().st_size == expected['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
    assert str(Path(__file__).resolve()) in go['inputs']
    output = WORK / 'validation_runs' / (args.profile + '_01')
    reservation = WORK / (args.profile + '_root_reservation_01.json')
    if output.exists() or reservation.exists():
        raise RuntimeError('profile already reserved; no retry under contract07')
    if args.profile == 'headless_zero_plane':
        prior = WORK / 'validation_runs/gui_policy_box_01/worker_receipt.json'
        record = json.loads(prior.read_text())
        assert record['execution_complete'] and record['failure'] is None
    command = [
        'rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT / 'scripts/play_d1_latest_rl.py'),
        '--actor', 'rl' if args.profile == 'gui_policy_box' else 'zero',
        '--terrain', 'box' if args.profile == 'gui_policy_box' else 'plane',
        '--validation-profile', args.profile, '--output', str(output),
        '--test-receipt', str(PURE), '--astra-go', str(GO),
    ]
    save(reservation, {
        'profile': args.profile, 'control_limit': 1200, 'normal_native_limit': 6000,
        'compiler_native_limit': 3, 'process_limit': 1,
        'argv': command, 'output': str(output), 'automatic_retry': False,
        'go_sha256': hashlib.sha256(GO.read_bytes()).hexdigest(),
    })
    def interrupted(_number, _frame):
        raise KeyboardInterrupt('root 07 validation interrupted')

    previous = signal.signal(signal.SIGTERM, interrupted)
    code, error, cleanup = None, None, None
    try:
        if args.profile == 'gui_policy_box':
            path = WORK.parent / 'gui_isolation/isolated_x11.py'
            spec = importlib.util.spec_from_file_location('private_x11_07', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            code = module.run(command, output)
        else:
            code = subprocess.run(command, check=False, cwd=ROOT).returncode
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        try:
            cleanup = cleanup_owned_worker(output)
        finally:
            signal.signal(signal.SIGTERM, previous)
            save(WORK / (args.profile + '_root_exit_01.json'),
                 {'exit_code': code, 'error': error, 'cleanup': cleanup,
                  'retry_permitted': False})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
