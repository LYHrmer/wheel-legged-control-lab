"""One cold import check under the candidate environment; no GUI or worker run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from worker30 import identity


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--go', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    go = json.loads(args.go.read_text())
    if go.get('bootstrap_only_candidate') is not True or go['decision'] != 'PENDING':
        raise RuntimeError('not the import-only review candidate')
    mismatches = [path for path, row in go['inputs'].items() if identity(path) != row]
    if mismatches:
        raise RuntimeError('source changed: ' + repr(mismatches))
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    environment = os.environ.copy()
    for key, value in go['runtime_environment'].items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    script = Path(__file__).with_name('bootstrap_check30.py')
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(script),
               '--go', str(args.go.resolve()), '--output', str(output)]
    started = time.monotonic()
    timed_out = False
    with (output / 'stdout.log').open('xb') as stream:
        child = subprocess.Popen(command, env=environment, cwd=go['repository'],
                                 stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        write(output / 'host_child.json', {'pid': child.pid, 'argv': command,
                                         'candidate_identity': identity(args.go)})
        try:
            code = child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=max(0.1, 90 - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    after = [path for path, row in go['inputs'].items() if identity(path) != row]
    receipt = {'schema': 'd1-c30-cold-bootstrap-host-v1', 'exit_code': code,
               'timed_out': timed_out, 'elapsed_s': time.monotonic() - started,
               'source_mismatches': after, 'child_reaped': child.poll() is not None,
               'retry_permitted': False, 'soft_s': 60, 'hard_s': 90}
    write(output / 'host_receipt.json', receipt)
    print(json.dumps(receipt))
    return 0 if code == 0 and not timed_out and not after else 1


if __name__ == '__main__':
    raise SystemExit(main())
