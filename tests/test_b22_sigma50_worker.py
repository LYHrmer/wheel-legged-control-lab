"""C50 contract/derivation and fake-state intervention tests; no real models."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker():
    return load("sigma50_worker_test", "scripts/worker_b22_sigma50.py")


def test_definition_import_never_imports_learning_or_physics():
    code = "import sys; sys.path.insert(0,'scripts'); import worker_b22_sigma50, run_b22_sigma50; assert not set(('torch','numpy','mujoco','stable_baselines3')) & set(sys.modules)"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, timeout=20)


def test_exact_derivation_rejects_wrong_or_ambiguous_seam(worker):
    derive = worker.contract50.ExactDerivation50("scripts/worker_b22_seed47.py")
    with pytest.raises(ValueError, match="count differs"):
        derive.expression("session_defaults42", "32768", "1", count=1)
    with pytest.raises(ValueError, match="old assignment"):
        derive.assignment("CONTRACT", "'not-the-frozen-contract'", "'anything'")


def test_worker_derivation_disables_old_cli_and_keeps_frozen_source(worker):
    tree, proof = worker.derived_worker50()
    assert proof["inverse_ast_exact"] is True
    assert len(proof["seams"]) == 12
    assert not any(isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) for node in tree.body)
    assert worker.contract50.identity(ROOT / "scripts/worker_b22_seed47.py")["sha256"] == worker.contract50.PINS["scripts/worker_b22_seed47.py"]


def test_six_fresh_arms_keep_original_one_environment_budget():
    parent = load("sigma50_parent_test", "scripts/run_b22_sigma50.py")
    stages = parent.stages42()
    assert [stage["name"] for stage in stages] == ["bootstrap", *["train_" + arm for arm in parent.contract50.ARMS]]
    assert sum(stage["controls"] for stage in stages) == 196608
    assert sum(stage["normal_native"] for stage in stages) == 983040
    assert sum(stage["compiler_native"] for stage in stages) == 12
    assert all(stage["controls"] == 32768 and stage["compiler_native"] == 2 for stage in stages[1:])
    assert {row["seed"] for row in parent.contract50.ARMS.values()} == {201051, 201061, 201071}
    assert str(parent.OUTPUT).endswith("b22_sigma50_20261024_01")
    assert parent.RESOURCE_POLICY["unit"] == "control50-wheel-train-20261024-01.service"


def test_session_bootstrap_and_every_seed_are_explicit(worker, monkeypatch):
    monkeypatch.setattr(worker.contract50, "dependency_hashes50", lambda: {"fixture": {"bytes": 0, "sha256": "0" * 64}})
    bootstrap = worker.session_defaults42("bootstrap", None)
    assert bootstrap["ppo_seed"] == bootstrap["seed"] == 201051
    assert bootstrap["controls"] == 0
    for arm, spec in worker.contract50.ARMS.items():
        session = worker.session_defaults42("train", arm)
        assert session["seed"] == session["ppo_seed"] == spec["seed"]
        assert session["controls"] == 32768
        assert session["model_limits"]["train"] == 32
        assert session["model_limits"]["evaluate_actions"] == 512
        assert session["sigma_offset_f32_hex"] == spec["offset_f32_hex"]


class Tensor:
    """Tiny mutable tensor fixture, with no framework or policy constructor."""

    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def add_(self, value):
        np.add(self.value, np.float32(value), out=self.value)


@pytest.mark.parametrize("label", ["baseline", "candidate"])
def test_actual_intervention_creates_new_audit_and_independent_proof(worker, monkeypatch, tmp_path, label):
    reader = load("sigma50_reader_intervention_test", "scripts/read_b22_sigma50.py")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float32=np.dtype("float32"), no_grad=contextlib.nullcontext,
        get_rng_state=lambda: Tensor(np.arange(16, dtype=np.uint8))))
    log_std = Tensor(np.linspace(-2.1, -1.9, 16, dtype=np.float32))
    actor_weights = {"0.weight": Tensor(np.ones((2, 2), dtype=np.float32))}
    action_weights = {"weight": Tensor(np.full((16, 2), 0.02, dtype=np.float32))}
    critic = Tensor(np.full((1, 2), 0.03, dtype=np.float32))
    optimizer = {"state": {0: {"step": Tensor(np.array(1635, dtype=np.float32)),
                               "exp_avg": Tensor(np.zeros(16, dtype=np.float32)),
                               "exp_avg_sq": Tensor(np.ones(16, dtype=np.float32))}},
                 "param_groups": [{"params": [0], "lr": 0.0003, "foreach": None}]}
    state = {"log_std": log_std, "value_net.weight": critic,
             **{"mlp_extractor.policy_net." + key: value for key, value in actor_weights.items()},
             **{"action_net." + key: value for key, value in action_weights.items()}}
    policy = SimpleNamespace(log_std=log_std, state_dict=lambda: state,
                             optimizer=SimpleNamespace(state_dict=lambda: optimizer),
                             mlp_extractor=SimpleNamespace(policy_net=SimpleNamespace(state_dict=lambda: actor_weights)),
                             action_net=SimpleNamespace(state_dict=lambda: action_weights))

    class Audit:
        def __init__(self, budget, seed, arm, target_kl):
            self.budget, self.ppo_seed, self.arm, self.target_kl = budget, seed, arm, target_kl
            self.train_calls = self.transitions = self.rollout_count = 0
            self.training_started = False
            self.initial_hashes = {}
            self.construction = {"stage": 2, "stage_counters_reset_only": True}

        def _hashes(self, model):
            arrays = {}
            trees = worker.capture_state50(model, arrays)
            used = set()
            return {key: reader.hash_state50(name, reader.decode_tree50(trees[name], arrays, used))
                    for key, name in reader.STATE_LABELS.items()}

        def snapshot_initial(self, model):
            assert not self.initial_hashes
            self.initial_hashes = self._hashes(model)

    model = SimpleNamespace(policy=policy)
    loaded = Audit("budget", 201051, "B", 0.03)
    loaded.snapshot_initial(model)
    loaded_snapshot = dict(loaded.initial_hashes)
    model.learning_audit = loaded
    parent = {"parent_hashes": loaded_snapshot, "parent_model_identity": {"sha256": "model", "bytes": 1},
              "parent_metadata_identity": {"sha256": "metadata", "bytes": 1}}
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return model, loaded, object(), parent

    class Writer:
        def commit_npz(self, path, arrays):
            with path.open("xb") as stream:
                np.savez(stream, **arrays)

        def commit_json(self, path, value):
            with path.open("x") as stream:
                json.dump(value, stream, allow_nan=False)

    arm = label + "_s51"
    spec = worker.contract50.ARMS[arm]
    session = {"reward_arm": arm, "ppo_seed": 201051, "sigma_multiplier": spec["sigma_multiplier"],
               "sigma_offset_f32_hex": spec["offset_f32_hex"]}
    returned, fresh, _clip, parentage = worker.load_warm_start50(
        original, SimpleNamespace(LearningAudit20=Audit), session, tmp_path, Writer())
    assert len(calls) == 1 and returned is model and fresh is not loaded and model.learning_audit is fresh
    assert loaded.initial_hashes == loaded_snapshot
    assert fresh.construction["stage_counters_reset_only"] is (label == "baseline")
    proof = json.loads((tmp_path / "initial_sigma50/proof.json").read_text())
    with np.load(tmp_path / "initial_sigma50/tensors.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    verified = reader.verify_initial_data50(proof, arrays, arm=arm, seed=201051,
        multiplier=spec["sigma_multiplier"], initial_hashes=fresh.initial_hashes, parent_hashes=loaded_snapshot)
    assert verified["actor_critic_adam_rng_unchanged"] is True
    assert parentage["initial_sigma50_proof_identity"] == worker.identity(tmp_path / "initial_sigma50/proof.json")
