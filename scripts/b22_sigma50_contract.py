"""Pure, versioned contract for the authorized initial-sigma experiment 50.

Importing this file does not import a numerical framework, scan the large source
closure, create a model, or start an experiment. All historical sources are read
only; the sole accepted environment difference belongs to the separate 50 audit.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "C50_B22_initial_sigma_v1"
SEEDS = (201051, 201061, 201071)
SEED = SEEDS[0]  # Compatibility name for the zero-construction bootstrap only.
ALPHA = 8
FEEDFORWARD = 3.9
CURRICULUM = "yaw_emphasis"
COMMAND_SEED = 201002
MEASUREMENT_SEED = 201003
SOURCE_AUDIT = ROOT / "results/b22_sigma50_source_audit_20261024_01/report.json"
MANIFEST = ROOT / "runtime/d1_b22/runtime_manifest_v27.json"
MANIFEST_SHA = "d3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09"
INITIAL_SCHEMA = "b22-sigma50-initial-intervention-v1"
OFFSET_F32_HEX = struct.pack("<f", math.log(0.5)).hex()
ARMS = {
    f"{label}_s{seed % 100}": {
        "alpha": ALPHA,
        "beta": 0,
        "curriculum": CURRICULUM,
        "seed": seed,
        "sigma_multiplier": multiplier,
        "offset_f32_hex": OFFSET_F32_HEX if multiplier == 0.5 else "00000000",
    }
    for seed in SEEDS
    for label, multiplier in (("baseline", 1.0), ("candidate", 0.5))
}
PINS = {
    "scripts/run_b22_seed47.py": "77fc77be60d195e1da5860a5f0a6dd1a31261aa8d063c36465054edc08e2468a",
    "scripts/worker_b22_seed47.py": "25856cf157244b78c436922f6388b7ba1a219aae0e64318faea97b6d579f1936",
    "scripts/read_b22_seed47.py": "712b346f4cdef9991ecb61c34af780d22c4f64602e5f49f65d269b2856faf688",
    "scripts/b22_seed47.py": "fe15bdd0f0b1f0b19db54f2d9b2596acc6416950f1249b128f41b8f0fe9ece70",
    "scripts/b22_yawtrain44.py": "5693b7e7a2284174d0f7e40ca6c40d80c5cc4b25ff020edb1ea7002e64b41110",
    "scripts/b22_reward46.py": "ad0b53cb0ed23654ba31809c71f486de0ce05cc5ac6147b91a1b6c964d0fc66b",
    "scripts/b22_tracking_reward42.py": "d05651d3e01bc5d4794721da3b403c95bfbc9be1364591ecbcddfbb59958e68e",
    "scripts/resource_guard42.py": "2fd0cbed467f36185a8fc442fe88f628a0566354404ea865260a6a76fb1f8fa2",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            "50 requires an absolute regular file: " + str(path))
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    after = path.stat()
    require(size == before.st_size and all(getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
            "50 source changed during hashing: " + str(path))
    return {"bytes": size, "sha256": digest.hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_audit50():
    helper = load_file("b22_sigma50_source_audit", ROOT / "scripts/audit_b22_sources50.py")
    return helper.verify_audit50(SOURCE_AUDIT, rehash=False)


def dependency_hashes50():
    report = source_audit50()
    require(report.get("status") == "COMPLETE", "50 source audit is incomplete")
    return copy.deepcopy(report["current_source_hashes"])


def pinned_inputs50():
    result = {}
    for name, expected in PINS.items():
        value = identity(ROOT / name)
        require(value["sha256"] == expected, "50 historical adapter changed: " + name)
        result[str(ROOT / name)] = value
    return result


class ExactDerivation50:
    """Named AST edits with exact old expressions and inverse restoration.

    This is deliberately not a string substitution utility. Call sites declare
    the enclosing function, exact expression, and expected occurrence count.
    """

    def __init__(self, relative):
        require(relative in PINS, "50 derivation source is not pinned")
        self.path = ROOT / relative
        require(identity(self.path)["sha256"] == PINS[relative], "50 derivation source changed")
        self.original = ast.parse(self.path.read_text())
        self.tree = copy.deepcopy(self.original)
        self.edits = []

    def _put(self, parent, field, index, value, label):
        old = getattr(parent, field) if index is None else getattr(parent, field)[index]
        self.edits.append((parent, field, index, old, label))
        if index is None:
            setattr(parent, field, value)
        else:
            getattr(parent, field)[index] = value

    def expression(self, function, old, new, *, count=1):
        scopes = [n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == function]
        require(len(scopes) == 1, "50 unique function required: " + function)
        target = ast.dump(ast.parse(old, mode="eval").body, include_attributes=False)
        replacement = ast.parse(new, mode="eval").body
        matches = []
        for parent in ast.walk(scopes[0]):
            for field, value in ast.iter_fields(parent):
                children = enumerate(value) if isinstance(value, list) else [(None, value)]
                for index, node in children:
                    if isinstance(node, ast.expr) and ast.dump(node, include_attributes=False) == target:
                        matches.append((parent, field, index))
        require(len(matches) == count, f"50 exact expression count differs: {function}: {old}")
        for parent, field, index in matches:
            self._put(parent, field, index, copy.deepcopy(replacement), function + ": " + old)

    def assignment(self, name, old, new):
        rows = [n for n in self.tree.body if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
        require(len(rows) == 1, "50 unique assignment required: " + name)
        require(ast.dump(rows[0].value, include_attributes=False)
                == ast.dump(ast.parse(old, mode="eval").body, include_attributes=False),
                "50 old assignment differs: " + name)
        self._put(rows[0], "value", None, ast.parse(new, mode="eval").body, "global " + name)

    def import_module(self, old, new, *, count):
        rows = [node for node in self.tree.body if isinstance(node, ast.ImportFrom) and node.module == old]
        require(len(rows) == count, "50 exact module import count differs: " + old)
        for node in rows:
            self._put(node, "module", None, new, "module import " + old)

    def remove_main_guard(self):
        rows = [(i, n) for i, n in enumerate(self.tree.body) if isinstance(n, ast.If)
                and ast.dump(n.test, include_attributes=False)
                == ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body, include_attributes=False)]
        require(len(rows) == 1, "50 expected one historical executable guard")
        index, _ = rows[0]
        self._put(self.tree, "body", index, ast.Pass(), "old CLI disabled in private derivation")

    def finish(self):
        result = copy.deepcopy(self.tree)
        for parent, field, index, old, _ in reversed(self.edits):
            if index is None:
                setattr(parent, field, old)
            else:
                getattr(parent, field)[index] = old
        require(ast.dump(self.tree, include_attributes=False) == ast.dump(self.original, include_attributes=False),
                "50 undeclared AST edit")
        return ast.fix_missing_locations(result), {
            "source": str(self.path), "source_identity": identity(self.path),
            "inverse_ast_exact": True, "seams": [row[4] for row in self.edits],
            "derived_ast_sha256": hashlib.sha256(ast.dump(result, include_attributes=False).encode()).hexdigest(),
        }


def curriculum_api50():
    require(identity(ROOT / "scripts/b22_yawtrain44.py")["sha256"] == PINS["scripts/b22_yawtrain44.py"],
            "50 frozen curriculum adapter changed")
    return load_file("b22_sigma50_yaw44", ROOT / "scripts/b22_yawtrain44.py")


def apply_curriculum(*args, **kwargs):
    return curriculum_api50().apply_curriculum(*args, **kwargs)


def derived_verifier_source(*args, **kwargs):
    return curriculum_api50().derived_verifier_source(*args, **kwargs)


def keep_record_consistent(*args, **kwargs):
    return curriculum_api50().keep_record_consistent(*args, **kwargs)
