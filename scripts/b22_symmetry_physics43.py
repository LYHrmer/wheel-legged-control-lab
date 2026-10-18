"""Shared fixed43 actor labels and existing recorder/scorer adapters.

Only identity constants/import names change in the pinned42 derived modules.
The original controller, native archive, schedules and scoring arithmetic remain.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'C43_B22_completed42_mirror_physics_v1'
ACTORS = ('zero', 'raw42', 'mirror43')
RECORD_SCHEMA = 'd1-b22-symmetry43-heldout-record-v1'

def require(value, message):
    if not value:
        raise ValueError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def api42():
    return load('symmetry43_worker_helpers', ROOT/'scripts/worker_b22_tracking_reward42.py')


def reader42():
    return load('symmetry43_reader_helpers', ROOT/'scripts/read_b22_tracking_reward42.py')


def adapt(kind, *, reading=False):
    api = reader42() if reading else api42()
    tree, original_proof = api.adapted_reader42(kind) if reading else api.adapt_module42(kind)
    before = ast.dump(tree, include_attributes=False)
    edits = []
    replacements = {'control': 'raw42', 'deficit8': 'mirror43',
                    'd1-b22-tracking42-heldout-record-v1': RECORD_SCHEMA}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            old = node.value
            new = replacements.get(old, old)
            if old.startswith('d1-b22-tracking42-'):
                new = old.replace('d1-b22-tracking42-', 'd1-b22-symmetry43-')
            if old != new:
                edits.append((node, 'value', old, new))
        elif isinstance(node, ast.ImportFrom) and node.module == 'b22_recipes42':
            edits.append((node, 'module', node.module, 'b22_recipes43'))
        elif isinstance(node, ast.alias) and node.name == 'b22_score42':
            edits.append((node, 'name', node.name, 'b22_score43'))
    for node, field, _, new in edits:
        setattr(node, field, new)
    derived = ast.dump(tree, include_attributes=False)
    code = compile(tree, str(ROOT/'scripts/b22_symmetry_physics43.py'), 'exec')
    for node, field, old, _ in edits:
        setattr(node, field, old)
    require(ast.dump(tree, include_attributes=False) == before, '43 changed arithmetic outside identity seams')
    return code, {'base42': original_proof, 'inverse_identity_edits_exact': True,
                  'derived_ast_sha256': hashlib.sha256(derived.encode()).hexdigest(),
                  'identity_edits': [[field, old, new] for _, field, old, new in edits]}


def recipes():
    api = api42()
    spec = json.loads(api.SPEC_PATH.read_text())
    # Evaluation recipe values are unchanged; training is never invoked here.
    code, proof = adapt('recipes')
    module = ModuleType('b22_recipes43')
    module.__file__ = str(ROOT/api.SOURCES['recipes'][0])
    module._spec42 = spec
    sys.modules[module.__name__] = module
    exec(code, module.__dict__)  # noqa: S102 -- pinned AST with identity-only changes
    return module, proof


def recorder():
    recipe, proof = recipes()
    code, heldout_proof = adapt('heldout')
    module = ModuleType('b22_heldout43')
    module.__file__ = str(ROOT/api42().SOURCES['heldout'][0])
    sys.modules[module.__name__] = module
    exec(code, module.__dict__)  # noqa: S102 -- original physical recorder, new actor labels
    require(module.ACTORS == ACTORS and module.RECORD_SCHEMA == RECORD_SCHEMA, '43 recorder labels differ')
    return recipe, module, {'recipes': proof, 'heldout': heldout_proof}


def saved_math():
    helper = reader42()
    helper.assert_saved_only42()
    sys.meta_path.insert(0, helper._NoExecution42())
    recipe, proof = recipes()
    spec = json.loads(api42().SPEC_PATH.read_text())
    modules, proofs = {}, {'recipes': proof}
    for kind, name in (('score', 'b22_score43'), ('eval', 'b22_eval43')):
        code, proof = adapt(kind, reading=True)
        module = ModuleType(name)
        module.__file__ = str(ROOT/helper.SOURCES[kind][0])
        module._spec42 = spec
        sys.modules[name] = module
        exec(code, module.__dict__)  # noqa: S102 -- original independent saved-data arithmetic
        module.CONTRACT_ID = CONTRACT
        modules[kind], proofs[kind] = module, proof
    return SimpleNamespace(recipes=recipe, score=modules['score'], reader=modules['eval'], proofs=proofs)


class Actor:
    """Same completed42 NumPy weights, optional reflection mean, exact calls."""

    def __init__(self, weights, *, symmetric, limit=20800):
        self.weights, self.limit = weights, limit
        self.attempts = self.returns = self.predictions = 0
        module = load('symmetry43_actor_math', ROOT/'src/wheel_legged_control/d1/policy_symmetry43.py')
        self.forward = module.SymmetricActor(self.raw) if symmetric else self.raw

    def raw(self, observation):
        import numpy as np
        x = np.asarray(observation)
        require(x.shape == (99,) and x.dtype == np.float32 and np.isfinite(x).all()
                and (abs(x) <= 5).all(), '43 prepared99 input required')
        require(self.attempts < self.limit, '43 actor call budget exhausted')
        self.attempts += 1
        w = self.weights
        x = np.tanh(x.reshape(1, 99) @ w['w1'].T + w['b1'])
        x = np.tanh(x @ w['w2'].T + w['b2'])
        action = np.clip(x @ w['w3'].T + w['b3'], np.float32(-1), np.float32(1)).reshape(16)
        require(action.shape == (16,) and np.isfinite(action).all(), '43 invalid action')
        self.returns += 1
        return action

    def predict(self, observation, deterministic=True):
        require(deterministic is True, '43 deterministic actor only')
        value = self.forward(observation)
        self.predictions += 1
        return value, None

    def counts(self):
        return {'attempts': self.attempts, 'returns': self.returns, 'predictions': self.predictions}
