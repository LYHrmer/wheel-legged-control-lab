"""Saved-data counterexamples for the independently read sigma intervention."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sigma50_reader_tests", ROOT / "scripts/read_b22_sigma50.py")
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


def encode(value, arrays):
    if isinstance(value, r.SavedArray50):
        key = f"v{len(arrays)}"
        arrays[key] = value.value.copy()
        return {"kind": value.kind, "key": key, "dtype": str(value.value.dtype), "shape": list(value.value.shape)}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[encode(k, arrays), encode(v, arrays)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__, "items": [encode(x, arrays) for x in value]}
    if isinstance(value, float):
        return {"kind": "float", "value": value.hex()}
    return {"kind": "none" if value is None else type(value).__name__, "value": value}


def fixture(multiplier=.5):
    def tensor(value, dtype="float32"):
        return r.SavedArray50("tensor", np.array(value, dtype=dtype))
    policy = {"log_std": tensor([-1.97 + i * .001 for i in range(16)]),
              "mlp_extractor.policy_net.0.weight": tensor([[.1, .2]]),
              "action_net.weight": tensor([[.3]]), "value_net.weight": tensor([[.4]])}
    before = {"policy": policy, "optimizer": {"state": {0: {"step": tensor(1300.),
               "exp_avg": tensor([.03]), "exp_avg_sq": tensor([.004])}}, "param_groups": [{"lr": .0001}]},
              "actor_mean": {"policy_net": {"0.weight": policy["mlp_extractor.policy_net.0.weight"]},
                             "action_net": {"weight": policy["action_net.weight"]}},
              "log_std": policy["log_std"],
              "rng": {"python": (3, (1, 2, 3), None),
                      "numpy": ("MT19937", r.SavedArray50("ndarray", np.array([4, 5], dtype="uint32")), 2, 0, 0.),
                      "torch": tensor([1, 2, 3], "uint8")}}
    after = copy.deepcopy(before)
    if multiplier == .5:
        after["policy"]["log_std"].value[:] += np.float32(math.log(.5))
    arrays = {}
    bh = {k: r.hash_state50(v, before[v]) for k, v in r.STATE_LABELS.items()}
    ah = {k: r.hash_state50(v, after[v]) for k, v in r.STATE_LABELS.items()}
    proof = {"schema": r.INITIAL_SCHEMA, "arm": "candidate_s51" if multiplier == .5 else "baseline_s51",
             "seed": 201051, "offset_f32_hex": struct.pack("<f", math.log(multiplier)).hex(),
             "source_parent_hashes": bh, "before_hashes": bh, "after_hashes": ah,
             "before": {k: encode(v, arrays) for k, v in before.items()},
             "after": {k: encode(v, arrays) for k, v in after.items()}}
    args = {"arm": proof["arm"], "seed": 201051, "multiplier": multiplier,
            "initial_hashes": ah, "parent_hashes": bh}
    return proof, arrays, args


def decoded(proof, arrays, phase):
    used = set()
    return {k: r.decode_tree50(v, arrays, used) for k, v in proof[phase].items()}


def resign_after(proof, arrays, args):
    """An attacker may rewrite receipt hashes; invariants must still reject."""
    values = decoded(proof, arrays, "after")
    hashes = {k: r.hash_state50(v, values[v]) for k, v in r.STATE_LABELS.items()}
    proof["after_hashes"] = hashes
    args["initial_hashes"] = hashes


@pytest.mark.parametrize("multiplier", [1., .5])
def test_only_registered_initial_shift_is_accepted(multiplier):
    proof, arrays, args = fixture(multiplier)
    result = r.verify_initial_data50(proof, arrays, **args)
    assert result["log_std_shape"] == [16]
    assert result["changed_policy_parameters"] == ([] if multiplier == 1 else ["log_std"])


def test_old_tensor_hash_keeps_t_tag_and_zero_dimensional_canonical_shape():
    array = np.array(2., dtype="float32")
    expected = hashlib.sha256(b"LoptimizerTfloat32(1,)" + struct.pack("<f", 2.)).hexdigest()
    assert r.hash_state50("optimizer", r.SavedArray50("tensor", array)) == expected
    assert r.hash_state50("optimizer", r.SavedArray50("ndarray", array)) != expected


@pytest.mark.parametrize("target", ["value_net.weight", "optimizer", "rng", "actor_mean"])
def test_resigned_receipts_cannot_hide_a_second_variable(target):
    proof, arrays, args = fixture()
    state = decoded(proof, arrays, "after")
    if target == "value_net.weight":
        state["policy"][target].value[0, 0] += 1
    elif target == "optimizer":
        state["optimizer"]["state"][0]["exp_avg"].value[:] = 0
    elif target == "rng":
        state["rng"]["torch"].value[0] += 1
    else:
        state["actor_mean"]["action_net"]["weight"].value[0, 0] += 1
    resign_after(proof, arrays, args)
    with pytest.raises(ValueError):
        r.verify_initial_data50(proof, arrays, **args)


def test_false_worker_boolean_and_unbound_tensors_are_rejected():
    proof, arrays, args = fixture()
    proof["verified"] = True
    with pytest.raises(ValueError, match="identity"):
        r.verify_initial_data50(proof, arrays, **args)
    proof.pop("verified")
    arrays["extra"] = np.zeros(1, dtype="float32")
    with pytest.raises(ValueError, match="unbound"):
        r.verify_initial_data50(proof, arrays, **args)


def test_offset_applied_to_policy_and_duplicate_is_still_recomputed():
    proof, arrays, args = fixture()
    state = decoded(proof, arrays, "after")
    state["policy"]["log_std"].value[:] += np.float32(.1)
    state["log_std"].value[:] += np.float32(.1)
    resign_after(proof, arrays, args)
    with pytest.raises(ValueError, match="registered float32"):
        r.verify_initial_data50(proof, arrays, **args)


def test_reader_ast_changes_one_predicate_and_retains_reload_kl_gradient_checks():
    path = ROOT / "runtime/d1_b22/research/continuation22/read_training22.py"
    original = ast.parse(path.read_text())
    derived, proof = r.replace_initial_predicate50(copy.deepcopy(original))
    function = next(n for n in derived.body if isinstance(n, ast.FunctionDef) and n.name == "_clip_and_learning")
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_verified_initial50"]
    assert len(calls) == 1 and proof["inverse_ast_exact"] is True
    call = calls[0]
    old = ast.parse('learned["initial_hashes"] == parent["parent_hashes"]', mode="eval").body
    for node in ast.walk(function):
        if isinstance(node, ast.BoolOp):
            node.values = [copy.deepcopy(old) if item is call else item for item in node.values]
    assert ast.dump(derived, include_attributes=False) == ast.dump(original, include_attributes=False)


def test_ast_gate_refuses_missing_or_duplicated_initial_predicate():
    tree = ast.parse("def _clip_and_learning():\n    pass\n")
    with pytest.raises(ValueError, match="predicate"):
        r.replace_initial_predicate50(tree)


def test_saved_only_admission_uses_registered_numpy_and_rejects_preloaded_foreign_origin(monkeypatch, tmp_path):
    registered = tmp_path / "registered" / "numpy.py"
    foreign = tmp_path / "foreign" / "numpy.py"
    module = SimpleNamespace(__file__=str(foreign))
    expected = {"bytes": 1, "sha256": "a" * 64}
    fake = SimpleNamespace(MANIFEST=tmp_path / "manifest.json",
        read_json=lambda _path: {"runtime_environment": {"PYTHONPATH": str(registered.parent)}},
        dependency_hashes50=lambda: {str(registered): expected}, identity=lambda _path: expected)
    real_import = builtins.__import__

    def import_hook(name, *args, **kwargs):
        if name == "numpy":
            assert sys.path[0] == str(registered.parent)
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(r, "contract50", lambda: fake)
    monkeypatch.setattr(builtins, "__import__", import_hook)
    with pytest.raises(ValueError, match="current source closure"):
        r.checked_numpy50()
    module.__file__ = str(registered)
    assert r.checked_numpy50() is module
