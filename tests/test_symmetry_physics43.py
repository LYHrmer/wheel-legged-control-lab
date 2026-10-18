import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helper():
    return load('test_physics43_helper', ROOT/'scripts/b22_symmetry_physics43.py')


def weights():
    rng = np.random.default_rng(4301)
    shapes = {'w1': (128, 99), 'b1': (128,), 'w2': (128, 128),
              'b2': (128,), 'w3': (16, 128), 'b3': (16,)}
    return {k: rng.normal(0, .05, shape).astype(np.float32) for k, shape in shapes.items()}


def test_raw_uses_the_previously_validated_batch_one_numpy_expression():
    h = helper()
    math36 = load('math36_test43', ROOT/'scripts/export_b22_numpy36.py')
    w = weights()
    actor = h.Actor(w, symmetric=False, limit=1)
    obs = np.linspace(-.9, .9, 99, dtype=np.float32)
    actual, state = actor.predict(obs)
    reference, _ = math36.numpy_action(obs, w, np)
    assert state is None and np.array_equal(actual, reference)
    assert actor.counts() == {'attempts': 1, 'returns': 1, 'predictions': 1}
    with pytest.raises(ValueError, match='budget'):
        actor.predict(obs)
    assert actor.counts() == {'attempts': 1, 'returns': 1, 'predictions': 1}


def test_mirror_two_calls_and_independent_projection_matches():
    h = helper()
    w = weights()
    sym = load('sym43_test', ROOT/'src/wheel_legged_control/d1/policy_symmetry43.py')
    obs = np.linspace(-.9, .9, 99, dtype=np.float32)
    raw = h.Actor(w, symmetric=False)
    projected = h.Actor(w, symmetric=True)
    first, _ = raw.predict(obs)
    second, _ = raw.predict(sym.mirror_observation(obs))
    expected = ((first.astype(float)+sym.mirror_action(second).astype(float))*.5).astype(np.float32)
    actual, _ = projected.predict(obs)
    assert np.array_equal(actual, expected)
    assert projected.counts() == {'attempts': 2, 'returns': 2, 'predictions': 1}


@pytest.mark.parametrize('bad', [np.zeros(98, np.float32), np.zeros(99), np.full(99, np.nan, np.float32)])
def test_bad_input_does_not_consume_inference(bad):
    actor = helper().Actor(weights(), symmetric=False)
    with pytest.raises(ValueError):
        actor.predict(bad)
    assert actor.counts() == {'attempts': 0, 'returns': 0, 'predictions': 0}


@pytest.mark.parametrize('kind,reading', [('recipes', False), ('heldout', False), ('score', True), ('eval', True)])
def test_pinned_derivation_only_changes_identity_not_scoring_or_physics(kind, reading):
    h = helper()
    code, proof = h.adapt(kind, reading=reading)
    assert code and proof['inverse_identity_edits_exact']
    assert proof['base42']['inverse_ast_exact']
    allowed = {'control', 'deficit8', 'b22_recipes42', 'b22_score42'}
    for _, before, _ in proof['identity_edits']:
        assert before in allowed or before.startswith('d1-b22-tracking42-')


def test_worker_budget_and_inference_are_new_zero_training_scope():
    run = load('run43_test', ROOT/'scripts/run_b22_symmetry_physics43.py')
    scope = run.scope()
    assert scope['controls_max'] == 2*600+3*9800
    assert scope['normal_native_max'] == 5*scope['controls_max']
    assert scope['numpy_actor_calls_max'] == (600+9800)*3
    assert scope['training'] == scope['ppo_load'] == scope['torch_load'] == 0
    # Keep the cold CLI declaration free of accidental model/environment calls.
    tree = ast.parse(Path(run.__file__).read_text())
    assert not any(isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names) for n in tree.body)
