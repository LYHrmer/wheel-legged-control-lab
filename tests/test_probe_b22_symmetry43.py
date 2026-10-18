import importlib.util
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
