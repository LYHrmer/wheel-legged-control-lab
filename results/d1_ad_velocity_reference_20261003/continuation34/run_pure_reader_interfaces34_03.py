"""Root-only finite reader fixtures, cold CLI and actual module inventory."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

C = Path(__file__).resolve().parent


def identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    start = time.monotonic()
    previous = json.loads((C / 'pure_reader_interfaces34_02/receipt.json').read_text())
    if previous['failure'] != 'Check 1 returned 1' or previous['source_before'] != previous['source_after']:
        raise RuntimeError('Earlier pure interface checks did not pass')
    consumed = previous['cumulative_pure_elapsed_s']
    directory = C / 'pure_reader_interfaces34_03'
    directory.mkdir(exist_ok=False)
    environment = os.environ.copy()
    for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'DISPLAY', 'XAUTHORITY'):
        environment.pop(key, None)
    environment.update(PYTHONPATH='/home/lyh/.local/lib/python3.10/site-packages:'+str(C / 'read34'),
        PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sources = [*sorted((C / 'read34').glob('*.py')), Path(__file__),
               C / 'reader_origins34_02.py', C / 'build_source_candidate34_02.py',
               C / 'launch_saved_reader34.py']
    before = {str(path): identity(path) for path in sources}
    commands = [
        ['rtk', 'proxy', '/home/lyh/wheel-legged-control-lab/.local-deps/bin/ruff',
         'check', '--no-cache', '--select', 'E9,F63,F7,F82', str(C / 'read34/test_reader34_interfaces_pure.py')],
        ['rtk', 'proxy', '/usr/bin/python3', '-B', '-m', 'unittest',
         'test_reader34_interfaces_pure.WorkerReaderInterface.test_c27_actual_side_index_is_one_based_and_contiguous', '-v'],
        ['rtk', 'proxy', '/usr/bin/python3', '-B', str(C / 'reader_origins34_02.py')],
    ]
    outcomes = []
    failure = None
    for index, command in enumerate(commands):
        began = time.monotonic()
        try:
            result = subprocess.run(command, env=environment, cwd=C,
                stdin=subprocess.DEVNULL, capture_output=True,
                timeout=max(.1, 60 - consumed - (began - start)))
            (directory / f'check_{index:02d}.stdout').write_bytes(result.stdout)
            (directory / f'check_{index:02d}.stderr').write_bytes(result.stderr)
            outcomes.append(dict(command=command, exit_code=result.returncode,
                                 elapsed_s=time.monotonic() - began))
            if result.returncode:
                failure = f'Check {index} returned {result.returncode}'
                break
        except BaseException as error:
            failure = repr(error)
            outcomes.append(dict(command=command, elapsed_s=time.monotonic()-began,
                                 failure=failure))
            break
    after = {str(path): identity(path) for path in sources}
    elapsed = time.monotonic() - start
    receipt = dict(schema='d1-c34-root-pure-reader-checks-v1',
        completed_utc=datetime.now(timezone.utc).isoformat(),
        prior_pure_elapsed_s=consumed, elapsed_s=elapsed,
        cumulative_pure_elapsed_s=consumed + elapsed, deadline_s=60,
        source_before=before, source_after=after, results=outcomes,
        failure=failure, no_engine_preload=True, new_model_calls=0,
        new_physics_steps=0, numerical_episode_reads=0, fits=0)
    with (directory / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(failure=failure, elapsed_s=elapsed,
        cumulative_pure_elapsed_s=consumed+elapsed, source_unchanged=before==after,
        exit_codes=[result.get('exit_code') for result in outcomes])))
    return 0 if not failure and before==after and consumed+elapsed <= 60 else 1


if __name__ == '__main__':
    raise SystemExit(main())
