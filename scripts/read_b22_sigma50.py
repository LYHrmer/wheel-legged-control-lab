"""Independent saved-only C50 reader; no Torch, actor or physics import.

The frozen 22 reader retains its PPO, KL, gradient, archive and final reload
checks. Its one parent-equality predicate is replaced by a tensor-level proof
of the registered initial log-standard-deviation intervention.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "C50_B22_initial_sigma_v1"
INITIAL_SCHEMA = "b22-sigma50-initial-intervention-v1"
STATE_LABELS = {"policy_state": "policy", "optimizer_state": "optimizer",
                "actor_mean_state": "actor_mean", "log_std_state": "log_std"}
EXTRA_PINS50 = {
    "scripts/probe_b22_symmetry43.py": "2413686d1c59f8baf999bfc40cc194e6cbd5607b27ca6a63b301326d95e62d05",
    "scripts/b22_symmetry_physics43.py": "967d31f8fac58d214bbcb3f052918a94659b5f865f1052d7d342ae23f05b4f04",
    "scripts/export_b22_numpy36.py": "8a89d8d1b82e59996bb09283050a40351a563fa19dbae37a4e17302893f466ce",
    "scripts/b22_numpy_backend38.py": "183053fe4f81bdfff1388587b143cd0890d35b7739ac3fced4425e0b4dbcbd62",
    "src/wheel_legged_control/d1/policy_symmetry43.py": "608cf390e5e6f8f11bdd4d2d9c98bdadf322059b132973678952821ed6168836",
    "results/b22_numpy36_20261018_01/inputs.f32.bin": "b1468a1d91007cecb3d8c8a706fb1f44fb2857819eb96f57fd9c72c5cfeda834",
    "runtime/d1_b22/research/rl11/rl16_learning_11.py": "837b1b7c814fcb174cc8f1d8ce0ad1bfcbf623e2cc4aba63ad36c3304f5ff673",
}
_BASE = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contract50():
    return load("_sigma50_reader_contract", ROOT / "scripts/b22_sigma50_contract.py")


def checked_numpy50():
    """Shared by full readback and zero-physics initial-proof admission."""
    contract = contract50()
    environment = contract.read_json(contract.MANIFEST)["runtime_environment"]
    sys.path[:0] = [entry for entry in environment["PYTHONPATH"].split(os.pathsep) if entry]
    import numpy as np

    path = Path(np.__file__).resolve()
    sources = contract.dependency_hashes50()
    require(sources.get(str(path)) == contract.identity(path),
            "C50 saved reader NumPy is not bound to the current source closure")
    return np


@dataclass(frozen=True)
class SavedArray50:
    kind: str
    value: object


def decode_tree50(node, arrays, used):
    """Decode a bounded typed tree without pickle or model/framework objects."""
    import numpy as np

    require(type(node) is dict and type(node.get("kind")) is str, "C50 typed node required")
    kind = node["kind"]
    if kind in ("tensor", "ndarray"):
        require(set(node) == {"kind", "key", "dtype", "shape"}, "C50 array node fields differ")
        key = node["key"]
        require(type(key) is str and key in arrays and key not in used, "C50 unique saved array required")
        used.add(key)
        value = arrays[key]
        require(isinstance(value, np.ndarray) and value.dtype.kind in "biuf"
                and value.flags.c_contiguous and np.isfinite(value).all()
                and type(node["dtype"]) is str and np.dtype(node["dtype"]) == value.dtype
                and type(node["shape"]) is list
                and all(type(x) is int and x >= 0 for x in node["shape"])
                and tuple(node["shape"]) == value.shape, "C50 saved array type/shape differs")
        return SavedArray50(kind, value)
    if kind in ("dict", "tuple", "list"):
        require(set(node) == {"kind", "items"} and type(node["items"]) is list,
                "C50 container node fields differ")
        if kind != "dict":
            values = [decode_tree50(x, arrays, used) for x in node["items"]]
            return tuple(values) if kind == "tuple" else values
        values = {}
        for item in node["items"]:
            require(type(item) is list and len(item) == 2, "C50 dict item differs")
            key = decode_tree50(item[0], arrays, used)
            require(type(key) in (str, int) and key not in values, "C50 unique str/int dictionary key required")
            values[key] = decode_tree50(item[1], arrays, used)
        return values
    require(set(node) == {"kind", "value"}, "C50 scalar node fields differ")
    value = node["value"]
    if kind == "none":
        require(value is None, "C50 null differs")
    elif kind in ("bool", "int", "str"):
        require(type(value) is {"bool": bool, "int": int, "str": str}[kind], "C50 scalar type differs")
    elif kind == "float":
        require(type(value) is str, "C50 hexadecimal float required")
        value = float.fromhex(value)
        require(math.isfinite(value), "C50 nonfinite scalar")
    elif kind == "bytes":
        require(type(value) is str, "C50 hexadecimal bytes required")
        value = bytes.fromhex(value)
    else:
        raise ValueError("C50 unsupported typed node: " + kind)
    return value


def _feed50(digest, value):
    """Independent reproduction of rl16_learning_11._feed, retaining T/A."""
    if value is None:
        digest.update(b"N")
    elif type(value) is bool:
        digest.update(b"B1" if value else b"B0")
    elif type(value) is int:
        digest.update(b"I" + repr(value).encode("ascii"))
    elif type(value) is float:
        digest.update(b"F" + struct.pack("<d", value))
    elif type(value) is str:
        digest.update(b"S" + value.encode("utf-8"))
    elif type(value) is bytes:
        digest.update(b"Y" + value)
    elif type(value) is dict:
        digest.update(b"D")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            digest.update(b"K" + repr(key).encode("utf-8"))
            _feed50(digest, value[key])
    elif type(value) in (tuple, list):
        digest.update(b"L")
        for item in value:
            _feed50(digest, item)
    elif isinstance(value, SavedArray50):
        import numpy as np
        array = np.ascontiguousarray(value.value)
        digest.update((b"T" if value.kind == "tensor" else b"A")
                      + str(array.dtype).encode("ascii") + repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    else:
        raise TypeError("C50 unsupported decoded state")


def hash_state50(label, value):
    digest = hashlib.sha256(b"L" + label.encode("utf-8"))
    _feed50(digest, value)
    return digest.hexdigest()


def same_state50(first, second):
    """Stricter than the old hash: list/tuple and scalar types also agree."""
    if type(first) is not type(second):
        return False
    if isinstance(first, SavedArray50):
        return (first.kind == second.kind and first.value.dtype == second.value.dtype
                and first.value.shape == second.value.shape
                and first.value.tobytes() == second.value.tobytes())
    if type(first) is dict:
        return first.keys() == second.keys() and all(same_state50(v, second[k]) for k, v in first.items())
    if type(first) in (tuple, list):
        return len(first) == len(second) and all(same_state50(a, b) for a, b in zip(first, second))
    if type(first) is float:
        return struct.pack("<d", first) == struct.pack("<d", second)
    return first == second


def verify_initial_data50(proof, arrays, *, arm, seed, multiplier, initial_hashes, parent_hashes):
    """Check raw evidence rather than worker-written success booleans."""
    import numpy as np

    required = {"schema", "arm", "seed", "offset_f32_hex", "source_parent_hashes",
                "before_hashes", "after_hashes", "before", "after"}
    require(type(proof) is dict and set(proof) == required and proof["schema"] == INITIAL_SCHEMA
            and proof["arm"] == arm and type(proof["seed"]) is int and proof["seed"] == seed,
            "C50 initial intervention identity differs")
    require(multiplier in (1.0, 0.5), "C50 unregistered sigma multiplier")
    offset = struct.pack("<f", 0.0 if multiplier == 1.0 else math.log(0.5))
    require(proof["offset_f32_hex"] == offset.hex(), "C50 initial float32 offset differs")
    used = set()
    states = {}
    for phase in ("before", "after"):
        require(type(proof[phase]) is dict and set(proof[phase]) == {*STATE_LABELS.values(), "rng"},
                "C50 complete policy/Adam/RNG snapshot required")
        states[phase] = {key: decode_tree50(value, arrays, used) for key, value in proof[phase].items()}
    require(used == set(arrays), "C50 unbound saved tensors")
    hashes = {phase: {key: hash_state50(label, states[phase][label])
                      for key, label in STATE_LABELS.items()} for phase in states}
    require(hashes["before"] == proof["before_hashes"] == proof["source_parent_hashes"] == parent_hashes
            and hashes["after"] == proof["after_hashes"] == initial_hashes,
            "C50 independently reconstructed initial/parent hashes differ")
    before, after = states["before"], states["after"]
    for phase in states.values():
        policy = phase["policy"]
        require(type(policy) is dict and "log_std" in policy, "C50 complete policy state required")
        log_std = phase["log_std"]
        require(isinstance(log_std, SavedArray50) and log_std.kind == "tensor"
                and log_std.value.dtype == np.dtype("float32") and log_std.value.shape == (16,)
                and same_state50(log_std, policy["log_std"]), "C50 policy/log_std duplicate binding differs")
        actor = {"policy_net": {key[len("mlp_extractor.policy_net."):]: value
                                 for key, value in policy.items() if key.startswith("mlp_extractor.policy_net.")},
                 "action_net": {key[len("action_net."):]: value
                                for key, value in policy.items() if key.startswith("action_net.")}}
        require(all(actor.values()) and same_state50(actor, phase["actor_mean"]),
                "C50 actor mean snapshot is not the policy's actor")
        require(type(phase["rng"]) is dict and set(phase["rng"]) == {"python", "numpy", "torch"},
                "C50 all three CPU RNG states required")
    require(before["policy"].keys() == after["policy"].keys()
            and all(same_state50(value, after["policy"][key]) for key, value in before["policy"].items()
                    if key != "log_std")
            and all(same_state50(before[key], after[key]) for key in ("optimizer", "actor_mean", "rng")),
            "C50 changed actor/critic/Adam/RNG outside the initial sigma intervention")
    prior = before["log_std"].value
    expected = (prior.copy() if multiplier == 1.0 else
                np.add(prior, np.frombuffer(offset, dtype="<f4")[0], dtype=np.float32))
    require(expected.tobytes() == after["log_std"].value.tobytes(),
            "C50 initial log_std differs from the registered float32 addition")
    return {"schema": "b22-sigma50-independent-initial-proof-v1", "arm": arm, "seed": seed,
            "offset_f32_hex": offset.hex(), "before_hashes": hashes["before"],
            "after_hashes": hashes["after"], "rng_sha256": hash_state50("rng", before["rng"]),
            "tensor_leaves_checked": len(used), "changed_policy_parameters": [] if multiplier == 1 else ["log_std"],
            "log_std_shape": [16], "actor_critic_adam_rng_unchanged": True}


def verify_initial50(run, session, learned, parent, *, documents, arrays):
    checked_numpy50()
    spec = contract50().ARMS[session["reward_arm"]]
    folder = Path(run) / "initial_sigma50"
    proof = documents(folder / "proof.json")
    saved = arrays(folder / "tensors.npz")
    base = base_reader50()
    require(base.identity(folder / "proof.json") == parent["initial_sigma50_proof_identity"],
            "C50 parentage intervention proof bytes differ")
    parent_metadata = Path(session["parentpaths"]["metadata"])
    require(base.identity(parent_metadata) == parent["parent_metadata_identity"],
            "C50 source parent metadata bytes differ")
    metadata = base.read_json(parent_metadata)
    require(metadata["training_receipt"]["status"] == "complete"
            and metadata["training_receipt"]["final_hashes"] == parent["parent_hashes"],
            "C50 complete source parent hashes differ")
    return verify_initial_data50(proof, saved, arm=session["reward_arm"], seed=spec["seed"],
                                 multiplier=spec["sigma_multiplier"],
                                 initial_hashes=learned["initial_hashes"], parent_hashes=parent["parent_hashes"])


def replace_initial_predicate50(tree):
    """Exactly one predicate in one frozen function; verify exact reversal."""
    original = copy.deepcopy(tree)
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_clip_and_learning")
    expected = ast.dump(ast.parse('learned["initial_hashes"] == parent["parent_hashes"]', mode="eval").body,
                        include_attributes=False)
    found = [(node, index, child) for node in ast.walk(target)
             for _field, value in ast.iter_fields(node) if _field == "values" and type(value) is list
             for index, child in enumerate(value) if isinstance(child, ast.Compare)
             and ast.dump(child, include_attributes=False) == expected]
    require(len(found) == 1, "C50 exact original initial-parent predicate not found")
    parent, index, previous = found[0]
    parent.values[index] = ast.parse("_verified_initial50(run, learned, parent)", mode="eval").body
    derived = copy.deepcopy(tree)
    parent.values[index] = previous
    require(ast.dump(tree, include_attributes=False) == ast.dump(original, include_attributes=False),
            "C50 undeclared training reader AST change")
    proof = {"function": "_clip_and_learning", "predicate": 'learned["initial_hashes"] == parent["parent_hashes"]',
             "replacement": "_verified_initial50(run, learned, parent)", "inverse_ast_exact": True,
             "derived_ast_sha256": hashlib.sha256(ast.dump(derived, include_attributes=False).encode()).hexdigest()}
    return ast.fix_missing_locations(derived), proof


def pairing50(numeric, gaussian, initial, parentage, session):
    base = base_reader50()
    spec = contract50().ARMS[session["reward_arm"]]
    require(session["seed"] == spec["seed"] and parentage["stage_seed"] == spec["seed"]
            and session["arm"] == parentage["arm"] == "B" and session["stage"] == parentage["stage"] == 2
            and parentage["actor_reinitialized"] is False and parentage["optimizer_reinitialized"] is False
            and set(initial) == set(base.INITIAL_FIELDS)
            and set(numeric) == set(base.NUMERIC_FIELDS) and set(gaussian) == set(base.GAUSSIAN_FIELDS)
            and all(value.shape[0] == 1024 for group in (numeric, gaussian) for value in group.values()),
            "C50 paired seed/course identity differs")
    # Changed actions and states are outcomes, never a cross-arm equality gate.
    return {"schema": "b22-sigma50-exogenous-pairing-v1", "seed": spec["seed"], "course_arm": "B",
            "command_seed": session["command_seed"], "measurement_seed": session["measurement_seed"],
            "source_episode_offset": session["source_episode_offset"],
            "initial_state": {key: base.array_identity42(value) for key, value in initial.items()},
            "parent_hashes": parentage["parent_hashes"],
            "first1024_trajectory_equality_required": False,
            "environment_randomness_scope": "fixed source_episode_index/episode_tick stream; PPO seed replicates"}


def base_reader50():
    global _BASE
    if _BASE is not None:
        return _BASE
    spec = contract50()
    spec.pinned_inputs50()
    base = load("_sigma50_frozen_reader47", ROOT / "scripts/read_b22_seed47.py")
    base.__file__ = __file__  # Bind this reader in the envelope; old source stays in closure.
    base.CONTRACT = CONTRACT
    base.ARMS = spec.ARMS
    base.train47.ARMS = spec.ARMS
    base.worker_api42 = lambda: load("_sigma50_reader_worker", ROOT / "scripts/worker_b22_sigma50.py")
    base.parent_api42 = lambda: load("_sigma50_reader_parent", ROOT / "scripts/run_b22_sigma50.py")
    base.pairing42 = pairing50
    original = base.adapted_reader42

    def adapted(kind):
        tree, proof = original(kind)
        if kind == "training":
            tree, intervention = replace_initial_predicate50(tree)
            proof = {"base47": proof, "sigma50_initial_predicate": intervention}
        return tree, proof

    base.adapted_reader42 = adapted
    _BASE = base
    return base


def load_math50(arm, worker_closure):
    base = base_reader50()
    base.assert_saved_only42()
    checked_numpy50()
    maths = base.load_math42(arm, worker_closure)
    train = maths.training
    proofs = {}
    episodes = []
    original_start = train._check_episode_start

    def start(folder, episode, source, course, offset, previous_end):
        import numpy as np

        result = original_start(folder, episode, source, course, offset, previous_end)
        recipe, schedule, reset, _state, _initial = result
        session = base.read_json(Path(folder).parent / "session.json")
        reset_seed = int(np.random.SeedSequence([session["measurement_seed"], source])
                         .generate_state(1, dtype=np.uint32)[0])
        provider_seed = int(np.random.default_rng(reset_seed).integers(0, 2**31))
        require(reset["episode_reset_seed"] == schedule["episode_reset_seed"] == reset_seed
                and reset["provider_measurement_seed"] == provider_seed
                and reset["actual_episode_metadata"]["measurement_seed"] == provider_seed,
                "C50 actual reset/measurement seed differs from the frozen source-index rule")
        require(episode == len(episodes), "C50 exogenous episode ledger skipped an episode")
        episodes.append({"episode_index": episode, "source_episode_index": source,
                         "raw_command_sha256": recipe.command_sha256,
                         "scheduled_ticks": len(recipe.raw_commands), "completed_ticks": 0,
                         "episode_reset_seed": reset_seed, "provider_measurement_seed": provider_seed,
                         "terminated": False, "truncated": False, "terminal_reason": None})
        return result

    original_numeric = train._check_numeric

    def numeric(row, *args):
        result = original_numeric(row, *args)
        episode = episodes[row["episode_index"]]
        require(row["episode_tick"] == episode["completed_ticks"], "C50 exogenous tick ledger differs")
        episode["completed_ticks"] += 1
        if row["terminated"] or row["truncated"]:
            episode.update(terminated=row["terminated"], truncated=row["truncated"],
                           terminal_reason=row["info"]["terminal_reason"])
        return result

    train._check_episode_start = start
    train._check_numeric = numeric

    def verified(run, learned, parent):
        session = base.read_json(Path(run) / "session.json")
        proofs["initial"] = verify_initial50(run, session, learned, parent,
                                               documents=train.document, arrays=train.arrays)
        return True

    train._verified_initial50 = verified
    original_ready = train._pairing_ready42

    def ready(*args):
        require("initial" in proofs, "C50 independent initial proof was not completed")
        require(sum(row["completed_ticks"] for row in episodes) == 32768,
                "C50 complete exogenous transition ledger required")
        return dict(original_ready(*args), rng_sha256=proofs["initial"]["rng_sha256"],
                    exogenous_episodes=episodes,
                    command_comparison="same full source-episode schedule; compare shared tick prefixes",
                    actual_episode_lengths_may_differ=True)

    train._pairing_ready42 = ready
    maths.proofs["initial_sigma50"] = proofs
    return maths


def read_stage42(release_path, stage_name, report_path=None, *, api=None):
    base = base_reader50()
    return base.read_stage42(release_path, stage_name, report_path,
                             api=api or base.parent_api42(), math_factory=load_math50)


def checked_saved_readback42(api, release_path, stage_name):
    return base_reader50().checked_saved_readback42(api, release_path, stage_name)


def required_inputs42():
    base = base_reader50()
    result = base.required_inputs42()
    for name, digest in EXTRA_PINS50.items():
        value = base.identity(ROOT / name)
        require(value["sha256"] == digest, "C50 frozen reader/evaluation input changed: " + name)
        result[str(ROOT / name)] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.release.is_absolute() and args.report.is_absolute(), "C50 absolute release/report required")
    value = read_stage42(args.release, args.stage, args.report)
    print(json.dumps({"status": value["status"], "stage": args.stage, "reader_model_calls": 0,
                      "reader_physics_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
