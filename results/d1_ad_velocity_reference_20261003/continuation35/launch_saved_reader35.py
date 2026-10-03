"""Own one reviewed C35 saved-data readback; no simulation or learned model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(path, *, deadline=None):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('saved reader overall 1200-second deadline')
            digest.update(block)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('saved reader overall 1200-second deadline')
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    review_path = args.review.resolve(strict=True)
    review = read(review_path)
    if review['decision'] != 'GO':
        raise RuntimeError('reader has not passed independent source review')
    for name in ('host_receipt.json', 'supervisor_receipt.json'):
        receipt = read(run/name)
        if name == 'host_receipt.json' and (receipt.get('source_mismatches') or not receipt.get('owned_no_orphans')):
            raise RuntimeError('physical owner did not close cleanly: '+name)
        if name == 'supervisor_receipt.json' and receipt.get('cleanup',{}).get('remaining'):
            raise RuntimeError('physical supervisor has remaining descendants')
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {**review['inputs'], str(review_path): identity(review_path),
               str(Path(__file__).resolve()): identity(__file__)}
    args.execution.mkdir(exist_ok=False)
    env = os.environ.copy()
    for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'DISPLAY', 'XAUTHORITY'):
        env.pop(key, None)
    env.update(PYTHONPATH='/home/lyh/.local/lib/python3.10/site-packages',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B', review['reader'],
               '--run', str(run), '--output', str(args.output.resolve()),
               '--source-go', str(review_path)]
    initial = dict(schema='d1-c35-saved-reader-execution-v1',
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   command=command, hard_limit_s=1200, sources=sources,
                   source_review_identity=identity(review_path))
    write(args.execution/'reservation.json', initial)
    process = None
    failure = None
    precheck_complete = False
    deadline = started+1200
    try:
        # Check every identity in the review, including inherited dependencies.
        for path, expected in sources.items():
            if identity(path, deadline=deadline) != expected:
                raise RuntimeError('reader source changed: '+path)
        precheck_complete = True
        with (args.execution/'stdout.log').open('xb') as stream:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            write(args.execution/'child.json', dict(pid=process.pid))
            # Reserve one minute for termination and final identity checks.
            process.wait(timeout=max(1, 1140-(time.monotonic()-started)))
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
    changed = []
    postcheck_complete = False
    postcheck_error = None
    postcheck_files = 0
    try:
        for path, expected in sources.items():
            if identity(path, deadline=deadline) != expected:
                changed.append(path)
            postcheck_files += 1
        postcheck_complete = True
        output_identity = identity(args.output, deadline=deadline) if args.output.is_file() else None
    except BaseException as error:
        postcheck_error = dict(type=type(error).__name__, message=str(error))
        if failure is None:
            failure = postcheck_error
        output_identity = None
    elapsed = time.monotonic()-started
    if elapsed > 1200 and failure is None:
        failure = dict(type='TimeoutError', message='reader overall deadline exceeded')
    result = dict(**initial, elapsed_s=elapsed,
                  exit_code=None if process is None else process.returncode,
                  failure=failure, source_mismatches=changed,
                  source_precheck_complete=precheck_complete,
                  source_postcheck_complete=postcheck_complete,
                  source_postcheck_files=postcheck_files,
                  source_postcheck_error=postcheck_error,
                  owned_child_reaped=process is None or process.returncode is not None,
                  output_path=str(args.output.resolve()),
                  output_identity=output_identity,
                  new_physics_steps=0, new_model_calls=0,
                  scope='reviewed saved-file readback; no robot or learned model execution')
    write(args.execution/'receipt.json', result)
    print(json.dumps({key: result[key] for key in ('exit_code', 'failure', 'elapsed_s',
                    'source_mismatches', 'output_identity')}), flush=True)
    return 0 if (result['exit_code'] == 0 and failure is None and not changed
                 and precheck_complete and postcheck_complete
                 and output_identity is not None and elapsed <= 1200) else 1


if __name__ == '__main__':
    raise SystemExit(main())
