"""Root-owned bounded test entry; forbids importing any physics runtime."""
import importlib.abc
import json
import os
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parent
ROOT = Path('/home/lyh/wheel-legged-control-lab')


class NoPhysics(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mujoco' or fullname.startswith(('mujoco.', 'wheel_legged_control')):
            self.attempts.append(fullname)
            raise ImportError('zero-call contract forbids '+fullname)


def main():
    counter = WORK/'pure_test_budget_01.json'
    budget = json.loads(counter.read_text()) if counter.exists() else {'attempted_rounds': 0, 'limit': 2}
    if budget['attempted_rounds'] >= 2:
        raise RuntimeError('pure test round budget exhausted')
    budget['attempted_rounds'] += 1
    counter.write_text(json.dumps(budget, indent=2)+'\n')
    round_index = budget['attempted_rounds']
    guard = NoPhysics()
    sys.meta_path.insert(0, guard)
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    sys.path.insert(0, str(ROOT))
    import pytest
    result = None
    try:
        result = int(pytest.main([str(ROOT/'tests/test_d1_single_step_archive_diagnostics.py'),
            '--junitxml='+str(WORK/f'pure_tests_round_{round_index:02d}.xml')]))
    finally:
        receipt = {'round': round_index, 'exit_code': result,
            'forbidden_import_attempts': guard.attempts,
            'mujoco_imported': any(k == 'mujoco' or k.startswith('mujoco.') for k in sys.modules),
            'new_control': 0, 'new_native': 0, 'new_static_mujoco': 0,
            'executor': 'root', 'implementation': 'root + actual gpt-6-sol; actual Astra directs/reviews'}
        with (WORK/f'pure_tests_round_{round_index:02d}_receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
    raise SystemExit(result)


if __name__ == '__main__':
    main()
