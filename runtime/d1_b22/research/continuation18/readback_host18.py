"""Bound a single clean saved-data readback, retaining failure and source identity."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import time
from worker18 import identity, write

C = Path(__file__).resolve().parent
W = C.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    receipt_path = output.with_name(output.stem+'_host_receipt.json')
    reserve_path = output.with_name(output.stem+'_reservation.json')
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(C/'read_eval18.py'),
               '--run', str(args.run.resolve(strict=True)), '--output', str(output)]
    write(reserve_path, {'argv': command, 'reader_identity': identity(C/'read_eval18.py'),
                         'wall_limit_s': 300, 'controls': 0, 'model_calls': 0,
                         'retry_permitted': False})
    env = dict(os.environ)
    old = json.loads((W/'rl11/training_run_01/session.json').read_text())
    for k, v in old['runtime_environment'].items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    env['PYTHONPATH'] += ':'+str(C)+':'+str(W/'continuation15')+':'+str(W/'continuation13')
    start = time.monotonic()
    proc = None
    failure = None
    try:
        with output.with_suffix('.log').open('xb') as log:
            proc = subprocess.Popen(command, env=env, cwd=C, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = proc.wait(timeout=295)
            if code != 0:
                raise RuntimeError('independent saved reader exited '+str(code))
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        result = {'argv': command, 'failure': failure,
                  'exit_code': None if proc is None else proc.returncode,
                  'elapsed_s': time.monotonic()-start, 'controls': 0, 'model_calls': 0,
                  'reader_identity': identity(C/'read_eval18.py'),
                  'output_identity': identity(output) if output.exists() else None}
        write(receipt_path, result)
        print(json.dumps(result), flush=True)
    return 0 if failure is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
