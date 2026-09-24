"""Prepay exactly one fixed drive-damping evaluation process; never retry."""
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.freeze.read_text())
    assert frozen['schema'] == 'drive-damping-evaluation-freeze-v1'
    expected = {'process_limit': 1, 'training_control_limit': 0,
                'evaluation_control_limit': 4800, 'normal_native_limit': 24000,
                'compiler_native_limit': 5, 'wallclock_limit_s': 600}
    assert all(type(frozen[key]) is int and frozen[key] == value
               for key, value in expected.items())
    assert frozen['astra_decision'] == 'GO'
    assert digest(frozen['contract_path']) == frozen['contract_sha256']
    assert str(Path(__file__).resolve()) in frozen['files']
    for name, row in frozen['files'].items():
        path = Path(name)
        assert path.is_absolute() and path.stat().st_size == row['bytes']
        assert digest(path) == row['sha256'], name
    prior = json.loads(Path(frozen['prior_boundary_closure']).read_text())
    assert prior['reservation_closed'] is True and prior['retry_permitted'] is False
    output = Path(frozen['output_directory'])
    budget = Path(frozen['budget_path'])
    assert not output.exists() and not budget.exists()
    argv = frozen['execution_argv']
    assert argv[:2] == ['rtk', 'proxy']
    assert str(output) in argv and str(args.freeze.resolve()) in argv
    activation = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'status': 'whole_drive_evaluation_reservation_activated_no_retry',
                  'freeze_sha256': digest(args.freeze),
                  'contract_sha256': frozen['contract_sha256'],
                  'execution_argv': argv, 'parent_pid': os.getpid(), **expected}
    output.mkdir()
    save(budget, activation)
    save(output / 'activation.json', activation)
    start = time.monotonic()
    exit_code = None
    error = None
    with (output / 'stdout.log').open('x') as out, (output / 'stderr.log').open('x') as err:
        try:
            child = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
            save(output / 'child_process.json',
                 {'pid': child.pid, 'parent_pid': os.getpid(), 'argv': argv})
            try:
                exit_code = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                error = 'timeout_whole_drive_evaluation_reservation_consumed'
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    exit_code = child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    exit_code = child.wait(timeout=3)
        except OSError as exc:
            error = type(exc).__name__ + ': ' + str(exc)
    mismatches = [name for name, row in frozen['files'].items()
                  if not Path(name).is_file() or digest(name) != row['sha256']]
    result = {'process_attempts': 1, 'exit_code': exit_code, 'error': error,
              'elapsed_seconds': time.monotonic() - start,
              'post_execution_hash_mismatches': mismatches, 'retry_permitted': False,
              'scope': 'new four-case controller intervention only; original interrupted counts remain unknown'}
    save(output / 'root_launcher_receipt.json', result)
    print(json.dumps(result, indent=2))
    return int(exit_code != 0 or error is not None or bool(mismatches))


if __name__ == '__main__':
    raise SystemExit(main())
