"""Root-owned, finite pure fixtures and CLI imports; no experiment execution."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

C = Path(__file__).resolve().parent
W = C.parent
R = Path('/home/lyh/wheel-legged-control-lab')


def identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    start = time.monotonic()
    old = json.loads((C / 'pure_controller_checks34_01/receipt.json').read_text())
    consumed = float(old['elapsed_s'])
    if old['passed'] is not True or abs(consumed - .249704983) > 1e-6:
        raise RuntimeError('Prior finite pure-check receipt differs')
    directory = C / 'pure_interfaces34_01'
    directory.mkdir(exist_ok=False)
    environment = os.environ.copy()
    for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'DISPLAY', 'XAUTHORITY'):
        environment.pop(key, None)
    parent = json.loads((W / 'continuation33/source_go33_02.json').read_text())
    old_paths = parent['runtime_environment']['PYTHONPATH'].split(':')
    if old_paths[0] != '/home/lyh/.local/lib/python3.10/site-packages':
        raise RuntimeError('Bootstrap site-packages first invariant differs')
    paths = [old_paths[0], str(C / 'root34'), str(C / 'sol34'), *old_paths[1:]]
    environment.update(PYTHONPATH=':'.join(dict.fromkeys(paths)),
        PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sources = [*sorted((C / 'root34').glob('*.py')), Path(__file__),
               C / 'reader_origins34.py', C / 'build_source_candidate34.py',
               C / 'launch_saved_reader34.py',
               W / 'continuation31/root31/task31.py',
               W / 'continuation31/root31/macro31.py',
               W / 'continuation30/root30/macro30.py']
    before = {str(path): identity(path) for path in sources}
    commands = [
        ['rtk', 'proxy', '/usr/bin/python3', '-B', '-m', 'unittest', 'discover',
         '-s', str(C / 'root34'), '-p', 'test_c34_pure.py', '-v'],
        ['rtk', 'proxy', str(R / '.local-deps/bin/ruff'), 'check', '--no-cache',
         '--select', 'E9,F63,F7,F82', *map(str, sources)],
        *[['rtk', 'proxy', '/usr/bin/python3', '-B', str(C / 'root34' / file),
           '--help'] for file in ('worker34.py', 'host34.py', 'build_request34.py')],
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
    receipt = dict(schema='d1-c34-root-pure-interface-checks-v1',
        completed_utc=datetime.now(timezone.utc).isoformat(),
        prior_controller_elapsed_s=consumed, elapsed_s=elapsed,
        cumulative_pure_elapsed_s=consumed + elapsed, deadline_s=60,
        source_before=before, source_after=after, results=outcomes,
        failure=failure, no_engine_preload=True, site_packages_first=True,
        new_model_calls=0, new_physics_steps=0, fits=0)
    with (directory / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(failure=failure, elapsed_s=elapsed,
        cumulative_pure_elapsed_s=consumed+elapsed, source_unchanged=before==after,
        exit_codes=[result.get('exit_code') for result in outcomes])))
    return 0 if not failure and before==after and consumed+elapsed <= 60 else 1


if __name__ == '__main__':
    raise SystemExit(main())
