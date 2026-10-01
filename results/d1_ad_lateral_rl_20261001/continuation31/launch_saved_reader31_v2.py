"""Run one reviewed saved-only readback with an exclusive receipt and hard limit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

P = Path(__file__).resolve().parent


def identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('training', 'evaluation'), required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    training = args.arm == 'training'
    review_path = P/('reader_source_review_31_v2.json' if training else 'eval_reader_source_review_31_v2.json')
    run = P/('train_01' if training else 'eval_01')
    review = read(review_path)
    if review['decision'] != 'GO':
        raise RuntimeError('saved reader is not independently reviewed')
    entry = Path(review['entry'])
    sources = {**review['inputs'], str(review_path): identity(review_path),
               str(Path(__file__).resolve()): identity(__file__)}
    for path, wanted in sources.items():
        if identity(path) != wanted:
            raise RuntimeError('saved reader input changed: '+path)
    for name in ('host_receipt.json', 'supervisor_receipt.json'):
        receipt = read(run/name)
        if receipt['exit_code'] != 0 or receipt['failure'] is not None:
            raise RuntimeError('runtime host is not closed successfully: '+name)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.execution.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env.pop('LD_LIBRARY_PATH', None)
    env.update(PYTHONPATH='/home/lyh/.local/lib/python3.10/site-packages',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(entry),
               '--run', str(run), '--output', str(args.output.resolve())]
    hard_s = 3600 if training else 1200
    started = time.monotonic()
    initial = dict(schema='d1-c31-saved-reader-execution-v1', arm=args.arm,
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   command=command, hard_limit_s=hard_s, sources=sources,
                   source_review_identity=identity(review_path))
    write(args.execution/'reservation.json', initial)
    failure = None
    process = None
    try:
        with (args.execution/'stdout.log').open('xb') as stream:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            write(args.execution/'child.json', dict(pid=process.pid))
            process.wait(timeout=hard_s)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
    changed = [path for path, wanted in sources.items() if identity(path) != wanted]
    result = dict(**initial, elapsed_s=time.monotonic()-started,
                  exit_code=None if process is None else process.returncode,
                  failure=failure, source_mismatches=changed,
                  owned_child_reaped=process is None or process.returncode is not None,
                  output_path=str(args.output.resolve()),
                  output_identity=identity(args.output) if args.output.is_file() else None,
                  new_physics_steps=0, new_model_calls=0,
                  scope='reviewed saved-file readback; no robot or learned model execution')
    write(args.execution/'receipt.json', result)
    print(json.dumps({key: result[key] for key in ('exit_code', 'failure', 'elapsed_s',
                                                  'source_mismatches', 'output_identity')}), flush=True)
    return 0 if result['exit_code'] == 0 and failure is None and not changed and args.output.is_file() else 1


if __name__ == '__main__':
    raise SystemExit(main())
