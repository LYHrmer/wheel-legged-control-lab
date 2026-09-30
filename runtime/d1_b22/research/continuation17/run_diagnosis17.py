"""Root-owned one-shot, saved-data-only diagnosis orchestration."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import time

C = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    with p.open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    start = time.monotonic()
    paths = [C/'astra_plan_17.md', Path(__file__),
             C/'yaw17/diagnose_saved_yaw17.py', C/'ramp17/diagnose_saved_ramp17.py']
    frozen = {str(p): digest(p) for p in paths}
    for p in paths:
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p))
    write(C/'diagnosis_reservation_17.json', {
        'source_sha256': frozen, 'controls': 0, 'native': 0, 'compiler': 0,
        'model_calls': 0, 'optimizer_steps': 0, 'per_reader_wall_limit_s': 180,
        'retry_permitted': False})
    environment = dict(os.environ)
    for k in ('LD_PRELOAD', 'LD_LIBRARY_PATH'):
        environment.pop(k, None)
    environment.update(PYTHONPATH='/home/lyh/.local/lib/python3.10/site-packages',
                       PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    receipts = []
    for name in ('yaw', 'ramp'):
        folder = C/(name+'17')
        target = folder/('saved_'+name+'_diagnosis_17.json')
        command = ['rtk', 'proxy', '/usr/bin/python3', '-B',
                   str(folder/('diagnose_saved_'+name+'17.py')), '--output', str(target)]
        begin = time.monotonic()
        failure = None
        code = None
        try:
            with (folder/'diagnosis_stdout_17.log').open('xb') as log:
                proc = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, timeout=180)
                code = proc.returncode
                if code:
                    raise RuntimeError('reader nonzero exit')
        except Exception as exc:
            failure = {'type': type(exc).__name__, 'message': str(exc)}
        receipts.append({'reader': name, 'command': command, 'exit_code': code,
                         'failure': failure, 'elapsed_s': time.monotonic()-begin,
                         'output_sha256': digest(target) if target.exists() else None})
    changed = [p for p, h in frozen.items() if digest(Path(p)) != h]
    result = {'schema': 'd1-stage17-saved-data-diagnosis-host-v1', 'readers': receipts,
              'source_sha256': frozen, 'changed_sources': changed,
              'elapsed_s': time.monotonic()-start, 'controls': 0, 'native': 0,
              'compiler': 0, 'model_calls': 0, 'optimizer_steps': 0,
              'complete': not changed and all(r['failure'] is None for r in receipts)}
    write(C/'diagnosis_17.json', result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
