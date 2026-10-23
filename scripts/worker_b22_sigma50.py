"""Independent C50 worker: exact historical owner, explicit initial-sigma hook.

Only definitions and pinned, standard-library adapters are loaded on import.
The parent owns every model, native-step, deadline and output reservation.
"""

from __future__ import annotations

import copy
import math
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b22_sigma50_contract as contract50


def derived_worker50():
    derive = contract50.ExactDerivation50("scripts/worker_b22_seed47.py")
    derive.assignment("WORKER", "ROOT/'scripts/worker_b22_seed47.py'", "ROOT/'scripts/worker_b22_sigma50.py'")
    derive.assignment("CONTRACT", "'C47_B22_seed_replication_v1'", repr(contract50.CONTRACT))
    derive.assignment("train47", "load_file('b22_seed47', ROOT/'scripts/b22_seed47.py')", "contract50")
    derive.expression("session_defaults42", "copy.deepcopy(config['source_hashes'])", "contract50.dependency_hashes50()")
    derive.expression("session_defaults42", "ROOT / 'docs/b22_seed47.md'", "ROOT / 'docs/b22_sigma50_preregistration.md'", count=2)
    derive.expression("bootstrap42", "'ffsse_s21'", "'candidate_s51'", count=2)
    derive.expression("bootstrap42", "recipes.T['ppo_seed_by_stage'][1]==201011",
                      "recipes.T['ppo_seed_by_stage'][1]==example['ppo_seed']")
    derive.expression("main", "ROOT/'scripts/run_b22_seed47.py'", "ROOT/'scripts/run_b22_sigma50.py'")
    derive.expression("main", "quota.receipt_json(args.output/'adapter42_proof.json',proof)",
                      "quota.receipt_json(args.output/'adapter42_proof.json', {**proof, 'outer_sigma50_derivation': DERIVATION50, 'namespace_overrides50': ['required_inputs42','session_defaults42','seed_learning_module42','train42']})")
    derive.remove_main_guard()
    return derive.finish()


_tree50, DERIVATION50 = derived_worker50()
exec(compile(_tree50, __file__, "exec"), globals())  # noqa: S102 -- pinned, inverse-checked named AST seams

ROOT = globals()["ROOT"]
CONTRACT = globals()["CONTRACT"]
identity = globals()["identity"]
require = globals()["require"]
main = globals()["main"]
_required_inputs47 = globals()["required_inputs42"]
_seed_learning47 = globals()["seed_learning_module42"]
_train47 = globals()["train42"]
_session_defaults47 = globals()["session_defaults42"]


def required_inputs42():
    values = _required_inputs47()
    values.update(contract50.pinned_inputs50())
    for name in ("scripts/b22_sigma50_contract.py", "scripts/worker_b22_sigma50.py",
                 "scripts/audit_b22_sources50.py"):
        values[str(ROOT / name)] = identity(ROOT / name)
    values[str(contract50.SOURCE_AUDIT)] = identity(contract50.SOURCE_AUDIT)
    # The audit verifier binds its complete JSONL ledger as well as this report.
    report = contract50.source_audit50()
    ledger = contract50.SOURCE_AUDIT.parent / "sources.jsonl"
    require(identity(ledger) == report["ledger_identity"], "50 audit ledger identity differs")
    values[str(ledger)] = identity(ledger)
    return values


def session_defaults42(mode, arm):
    value = _session_defaults47(mode, arm)
    selected = contract50.ARMS[arm] if mode == "train" else contract50.ARMS["candidate_s51"]
    value.update(
        sigma_multiplier=selected["sigma_multiplier"] if mode == "train" else None,
        sigma_offset_f32_hex=selected["offset_f32_hex"] if mode == "train" else None,
        initial_intervention_schema=contract50.INITIAL_SCHEMA,
        seed=selected["seed"], ppo_seed=selected["seed"],
        run_id="sigma50_" + ("train_" + arm if mode == "train" else mode),
        phase="sigma50_" + mode,
    )
    return value


def encode_state50(value, arrays):
    """Serialize actual state without pickle; tensor/array types remain distinct."""
    import numpy as np

    if value is None:
        return {"kind": "none", "value": None}
    if isinstance(value, (bool, np.bool_)):
        return {"kind": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"kind": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        require(math.isfinite(float(value)), "50 nonfinite scalar state")
        return {"kind": "float", "value": float(value).hex()}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "value": value.hex()}
    if isinstance(value, dict):
        require(all(type(key) in (int, str) for key in value), "50 unsupported state dictionary key")
        return {"kind": "dict", "items": [[encode_state50(key, arrays), encode_state50(value[key], arrays)]
                for key in sorted(value, key=lambda item: (type(item).__name__, repr(item)))]}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [encode_state50(item, arrays) for item in value]}
    tensor = hasattr(value, "detach") and hasattr(value, "cpu")
    require(tensor or isinstance(value, np.ndarray), "50 unsupported state value: " + type(value).__name__)
    array = np.ascontiguousarray(value.detach().cpu().numpy() if tensor else value).copy()
    require(array.dtype.kind in "biuf" and np.isfinite(array).all(), "50 invalid numeric state array")
    key = f"array_{len(arrays):06d}"
    arrays[key] = array
    return {"kind": "tensor" if tensor else "ndarray", "key": key,
            "dtype": str(array.dtype), "shape": list(array.shape)}


def capture_state50(model, arrays):
    import numpy as np
    import torch

    policy = model.policy
    values = {
        "policy": policy.state_dict(), "optimizer": policy.optimizer.state_dict(),
        "actor_mean": {"policy_net": policy.mlp_extractor.policy_net.state_dict(),
                       "action_net": policy.action_net.state_dict()},
        "log_std": policy.log_std.detach(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
    }
    return {name: encode_state50(value, arrays) for name, value in values.items()}


def load_warm_start50(original, learning, session, output, writer, *args, **kwargs):
    """Verify the exact parent first, then create an honest new initial audit."""
    import numpy as np
    import torch

    model, loaded_audit, clip, parentage = original(*args, **kwargs)
    require(session["reward_arm"] in contract50.ARMS, "50 initial intervention arm unknown")
    selected = contract50.ARMS[session["reward_arm"]]
    require(session["ppo_seed"] == selected["seed"] and session["sigma_multiplier"] == selected["sigma_multiplier"]
            and session["sigma_offset_f32_hex"] == selected["offset_f32_hex"], "50 intervention differs from registered arm")
    require(loaded_audit.initial_hashes == parentage["parent_hashes"] and not loaded_audit.training_started
            and loaded_audit.train_calls == loaded_audit.transitions == loaded_audit.rollout_count == 0,
            "50 intervention must follow verified parent and precede all learning")
    policy = model.policy
    require(policy.log_std.dtype == torch.float32 and tuple(policy.log_std.shape) == (16,),
            "50 exactly sixteen float32 log_std parameters required")
    arrays = {}
    before = capture_state50(model, arrays)
    before_hashes = loaded_audit._hashes(model)
    old_std = policy.log_std.detach().cpu().numpy().copy()
    offset = struct.unpack("<f", bytes.fromhex(selected["offset_f32_hex"]))[0]
    if selected["sigma_multiplier"] == 0.5:
        with torch.no_grad():
            policy.log_std.add_(offset)
    expected = np.add(old_std, np.float32(offset), dtype=np.float32) if offset else old_std
    require(policy.log_std.detach().cpu().numpy().tobytes() == expected.tobytes(),
            "50 actual initial float32 offset differs")
    after = capture_state50(model, arrays)
    after_hashes = loaded_audit._hashes(model)
    require(before_hashes["optimizer_state"] == after_hashes["optimizer_state"]
            and before_hashes["actor_mean_state"] == after_hashes["actor_mean_state"],
            "50 initial intervention changed Adam or actor mean")
    # The independent reader reconstructs every policy/critic/Adam/RNG leaf;
    # these local checks are additional guards, never substitutes for readback.
    audit = learning.LearningAudit20(loaded_audit.budget, session["ppo_seed"], loaded_audit.arm, loaded_audit.target_kl)
    audit.construction = copy.deepcopy(loaded_audit.construction)
    audit.construction.update(stage_counters_reset_only=offset == 0,
                              initial_sigma50={"schema": contract50.INITIAL_SCHEMA,
                                               "offset_f32_hex": selected["offset_f32_hex"],
                                               "sigma_multiplier": selected["sigma_multiplier"],
                                               "applied_once_before_first_rollout": True})
    audit.clip_audit = clip
    model.learning_audit = audit
    audit.snapshot_initial(model)
    require(audit.initial_hashes == after_hashes, "50 fresh audit lost actual initial state")
    folder = Path(output) / "initial_sigma50"
    folder.mkdir(exist_ok=False)
    writer.commit_npz(folder / "tensors.npz", arrays)
    proof = {
        "schema": contract50.INITIAL_SCHEMA,
        "arm": session["reward_arm"], "seed": session["ppo_seed"],
        "offset_f32_hex": selected["offset_f32_hex"],
        "source_parent_hashes": copy.deepcopy(parentage["parent_hashes"]),
        "before_hashes": before_hashes, "after_hashes": after_hashes,
        "before": before, "after": after,
    }
    writer.commit_json(folder / "proof.json", proof)
    writer.commit_json(folder / "receipt.json", {
        "schema": "b22-sigma50-initial-worker-receipt-v1", "execution_contract_id": CONTRACT,
        "proof_identity": identity(folder / "proof.json"),
        "sigma_multiplier": selected["sigma_multiplier"],
        "tensors_identity": identity(folder / "tensors.npz"),
        "loaded_audit_construction": copy.deepcopy(loaded_audit.construction),
        "initial_audit_construction": copy.deepcopy(audit.construction),
        "source_parent_model_identity": parentage["parent_model_identity"],
        "source_parent_metadata_identity": parentage["parent_metadata_identity"],
        "additional_model_loads": 0, "additional_optimizer_steps": 0,
        "additional_environment_steps": 0,
    })
    parentage = dict(parentage, initial_sigma50_proof_identity=identity(folder / "proof.json"))
    return model, audit, clip, parentage


def seed_learning_module42(session):
    learning = _seed_learning47(session)
    original = learning.load_warm_start20
    context = _initial_context50
    require(context["session"] is session, "50 warm-start context differs")

    def wrapped(*args, **kwargs):
        return load_warm_start50(original, learning, session, context["output"], context["writer"], *args, **kwargs)

    learning.load_warm_start20 = wrapped
    return learning


def train42(session, runtime, guard, env, output, writer, counter):
    global _initial_context50
    _initial_context50 = {"session": session, "output": output, "writer": writer}
    try:
        return _train47(session, runtime, guard, env, output, writer, counter)
    finally:
        _initial_context50 = None


_initial_context50 = None

if __name__ == "__main__":
    raise SystemExit(main())
