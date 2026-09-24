"""At most two prepaid rounds, each at most twelve pure body-P tests."""
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
                    or fullname in ('drive_damping_controller_04', 'drive_damping_transfer_env_04', 'drive_damping_transfer_env_05', 'body_speed_controller_06', 'body_speed_transfer_env_06')):
                blocked.append(fullname)
                raise ImportError('pure drive test import barrier: ' + fullname)

    sys.meta_path.insert(0, Barrier())
    names = ['body_speed_math_06.py', 'body_speed_controller_06.py',
             'body_speed_validator_06.py', 'body_speed_transfer_env_06.py',
             'test_body_common_p_opus_01.py', 'test_body_chain_root_01.py',
             'body_validator_fixture_02.json.gz', Path(__file__).name]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    paths = [D / name for name in names] + [D.parent / 'stability_20260924_drive_damping01' / name for name in ('drive_damping_math_04.py', 'drive_damping_validator_04.py')]
    before = {str(path): digest(path) for path in paths}
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
        code = pytest.main(['-q', str(D / 'test_body_common_p_opus_01.py'),
                            str(D / 'test_body_chain_root_01.py'),
                            '-p', 'no:cacheprovider'], plugins=[counts])
    with (D / (prefix + '.log')).open('x') as stream:
        stream.write(out.getvalue())
    after = {str(path): digest(path) for path in paths}
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
