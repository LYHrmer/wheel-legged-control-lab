"""Root-only bounded arithmetic tests and clean reader import check."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import time

C = Path(__file__).resolve().parent
W = C.parent


def main():
    paths = sorted(C.glob('*.py'))
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:
        ast.parse(p.read_text(), filename=str(p))
    env = dict(os.environ)
    old = json.loads((W/'rl11/training_run_01/session.json').read_text())
    for k, v in old['runtime_environment'].items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    for k in ('LD_PRELOAD', 'LD_LIBRARY_PATH'):
        env.pop(k, None)
    env['PYTHONPATH'] += ':'+str(C)+':'+str(W/'continuation15')+':'+str(W/'continuation13')
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    commands = [
        ['rtk', 'proxy', '/usr/bin/python3', '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(C/'test_control17.py')],
        ['rtk', 'proxy', '/usr/bin/python3', '-B', str(C/'offline_floor17.py'), '--probe-import', '--output', str(C/'pure_reader_import_17.json')],
        ['rtk', 'proxy', '/usr/bin/python3', '-B', '-m', 'ruff', 'check', '--select', 'E9,F63,F7,F82', *map(str, paths)],
    ]
    receipts = []
    for i, cmd in enumerate(commands):
        started = time.monotonic()
        failure = None
        code = None
        try:
            with (C/f'pure_check_{i:02d}.log').open('xb') as stream:
                proc = subprocess.run(cmd, env=env, cwd=C, stdin=subprocess.DEVNULL,
                                      stdout=stream, stderr=subprocess.STDOUT, timeout=60)
                code = proc.returncode
        except Exception as exc:
            failure = repr(exc)
        receipts.append({'argv': cmd, 'exit_code': code, 'failure': failure,
                         'elapsed_s': time.monotonic()-started})
    changed = [s for s, h in frozen.items() if hashlib.sha256(Path(s).read_bytes()).hexdigest() != h]
    result = {'source_sha256': frozen, 'checks': receipts, 'changed_sources': changed,
              'controls': 0, 'native_steps': 0, 'model_calls': 0, 'optimizer_steps': 0,
              'passed': not changed and all(r['exit_code'] == 0 and r['failure'] is None for r in receipts)}
    with (C/'pure_tests_receipt_17.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
