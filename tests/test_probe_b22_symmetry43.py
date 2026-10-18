import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('probe43_test', ROOT/'scripts/probe_b22_symmetry43.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_numeric_scope_covers_two_calls_and_no_physics():
    scope = probe.scope()
    assert scope['sb3_predict_max'] == scope['numpy_predict_max'] == 2*scope['symmetric_predict_max'] == 384
    assert scope['physics_steps'] == scope['environment_constructions'] == scope['training_calls'] == 0
    assert scope['retry_permitted'] is scope['closed_loop_qualification'] is False


@pytest.mark.parametrize('change', ['shape', 'dtype', 'nan', 'drift'])
def test_numerical_parity_rejects_incomplete_or_wrong_outputs(change):
    reference = np.zeros((384, 16), np.float32)
    actual = reference.copy()
    if change == 'shape':
        actual = actual[:-1]
    elif change == 'dtype':
        actual = actual.astype(np.float64)
    elif change == 'nan':
        actual[0, 0] = np.nan
    else:
        actual[0, 0] = 1.1e-6
    with pytest.raises(ValueError):
        probe.compare(reference, actual, np)


def test_parity_uses_max_error_over_every_row_and_channel():
    reference = np.zeros((384, 16), np.float32)
    actual = reference.copy()
    actual[-1, -1] = 9e-7
    assert probe.compare(reference, actual, np) == float(actual[-1, -1])


def test_cli_plan_ignores_unrelated_regular_scripts_package(tmp_path):
    package = tmp_path/'scripts'
    package.mkdir()
    (package/'__init__.py').write_text('raise RuntimeError("foreign ROS scripts package")\n')
    result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/probe_b22_symmetry43.py')],
                            cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(tmp_path)},
                            text=True, capture_output=True, check=True, timeout=10)
    assert json.loads(result.stdout)['status'] == 'PLAN_ONLY'


def test_actual_cold_symmetry_loader_does_not_import_simulator():
    code = '''import importlib.util,sys
s=importlib.util.spec_from_file_location('probe43','scripts/probe_b22_symmetry43.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class NoSimulator:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.partition('.')[0] in ('mujoco','engine_binding','torch','stable_baselines3'):
   raise ImportError('forbidden '+fullname)
sys.meta_path.insert(0,NoSimulator())
p=m.load_symmetry()
import numpy as np
assert p.mirror_observation(np.zeros(99,np.float32)).shape==(99,)
assert not any(x in sys.modules for x in ('mujoco','engine_binding','torch','stable_baselines3'))
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                   capture_output=True, text=True, check=True, timeout=10)
