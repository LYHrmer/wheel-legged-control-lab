"""At most two prepaid rounds, each at most twelve pure controller-record tests."""
import argparse
import contextlib
import hashlib
import importlib.abc
import io
import json
import os
import sys
from pathlib import Path

D = Path(__file__).resolve().parent


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=(1, 2), required=True)
    parser.add_argument('--math-tests', required=True, choices=(
        'test_drive_damping_math_opus_01.py', 'test_drive_damping_math_sol_01.py'))
    args = parser.parse_args()
    assert not os.environ.get('LD_PRELOAD')
    assert os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
    prefix = f'pure_test_round_{args.round}'
    if args.round == 2:
        assert (D / 'pure_test_round_1_receipt.json').is_file()
    blocked = []

    class Barrier(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if (fullname.split('.')[0] in ('mujoco', 'gymnasium', 'torch', 'stable_baselines3')
                    or fullname.startswith(('wheel_legged_control.', 'scripts.'))
                    or fullname in ('drive_damping_controller_04', 'drive_damping_transfer_env_04')):
                blocked.append(fullname)
                raise ImportError('pure drive test import barrier: ' + fullname)

    sys.meta_path.insert(0, Barrier())
    names = ['drive_damping_math_04.py', 'drive_damping_validator_04.py',
             'drive_damping_controller_04.py', args.math_tests,
             'test_drive_stage_records_root_01.py', Path(__file__).name]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    before = {str(D / name): digest(D / name) for name in names}
    save(D / (prefix + '_activation.json'),
         {'round': args.round, 'max_cases': 12, 'engine_imports_forbidden': True,
          'input_sha256': before})
    import pytest

    class Counts:
        def __init__(self):
            self.collected = 0
            self.rows = []

        def pytest_collection_modifyitems(self, items):
            self.collected = len(items)
            if self.collected > 12:
                raise pytest.UsageError('pure drive test case cap exceeded')

        def pytest_runtest_logreport(self, report):
            if report.when == 'call':
                self.rows.append({'nodeid': report.nodeid, 'outcome': report.outcome})

    counts = Counts()
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = pytest.main(['-q', str(D / names[3]), str(D / names[4]),
                            '-p', 'no:cacheprovider'], plugins=[counts])
    with (D / (prefix + '.log')).open('x') as stream:
        stream.write(out.getvalue())
    after = {str(D / name): digest(D / name) for name in names}
    engine_loaded = any(name == 'mujoco' or name.startswith('mujoco.') for name in sys.modules)
    passed = (code == 0 and before == after and not blocked and not engine_loaded
              and len(counts.rows) == counts.collected == 12
              and all(row['outcome'] == 'passed' for row in counts.rows))
    receipt = {'round': args.round, 'exit_code': int(code), 'collected': counts.collected,
               'cases': counts.rows, 'blocked_import_attempts': blocked,
               'engine_module_loaded': engine_loaded, 'input_sha256': before,
               'inputs_unchanged': before == after, 'passed': passed}
    save(D / (prefix + '_receipt.json'), receipt)
    print(out.getvalue())
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in ('cases', 'input_sha256')}))
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
