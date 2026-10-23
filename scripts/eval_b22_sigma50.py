"""One complete C50 final -> NumPy parity -> all six tasks and both floors.

Each mode is a separate, externally supervised process. This driver never
starts another worker, retries a stage, chooses a seed, or replaces a policy.
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
import signal
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "C50_B22_initial_sigma_v1"
ACTORS = ("zero", "raw50", "mirror50")
POLICIES = ACTORS[1:]
RECORD_SCHEMA = "d1-b22-sigma50-heldout-record-v1"
FIXTURE = ROOT / "results/b22_numpy36_20261018_01/inputs.f32.bin"
FLOOR_SEED = 201090
CONTROL_CAP = 30600
CASE_CONTROLS = 29400


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contract():
    return load("_sigma50_eval_contract", ROOT / "scripts/b22_sigma50_contract.py")


def training_reader():
    return load("_sigma50_eval_training_reader", ROOT / "scripts/read_b22_sigma50.py")


def parent():
    return load("_sigma50_eval_parent", ROOT / "scripts/run_b22_sigma50.py")


def identity(path):
    return contract().identity(Path(path))


def read(path):
    return contract().read_json(path)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inputs50(release, stage):
    api = parent()
    reader = training_reader()
    bound = api.validate_release42(release)
    saved, saved_path = reader.checked_saved_readback42(api, release, stage)
    require(saved.get("training_valid") is True and saved.get("archive_valid") is True
            and saved.get("learning_valid") is True, "C50 complete independently read final required")
    spec = api.stage_by_name42(stage)
    require(spec["mode"] == "train", "C50 final must come from a registered training stage")
    run = api.OUTPUT / stage
    final = reader.base_reader50().final_checkpoint_inputs42(run)
    metadata = read(run / "final_checkpoint/final_metadata.json")
    require(metadata["execution_contract_id"] == CONTRACT
            and metadata["training_receipt"]["status"] == "complete"
            and metadata["training_receipt"]["actual"]["num_timesteps"] == 32768,
            "C50 complete fixed-budget final metadata required")
    sources = dict(bound["inputs"])
    for path, entry in reader.required_inputs42().items():
        require(sources.get(path) == entry, "C50 eval helper absent from current source closure: " + path)
    for path in (Path(__file__), ROOT / "scripts/read_b22_sigma50.py"):
        require(sources.get(str(path)) == identity(path), "C50 current eval/reader not in release closure")
    return {"training_stage": stage, "training_run": str(run), "training_arm": spec["reward_arm"],
            "training_seed": contract().ARMS[spec["reward_arm"]]["seed"],
            "training_coverage_valid": saved["coverage_valid"], "training_coverage": saved["coverage"],
            "training_readback_path": str(saved_path), "training_readback_identity": identity(saved_path),
            "final_files": final, "checkpoint_sha256": identity(run / "final_checkpoint/final_model.zip")["sha256"],
            "source_hashes": sources, "fixture_identity": identity(FIXTURE)}


def prepare50(release, stage, output):
    require(release.is_absolute() and output.is_absolute() and not output.exists(),
            "C50 fresh absolute evaluation output required")
    require(output.parent == parent().OUTPUT and output.name == "eval_" + stage.removeprefix("train_"),
            "C50 fixed per-model evaluation output required")
    values = inputs50(release, stage)
    output.mkdir(exist_ok=False)
    plan = {"schema": "b22-sigma50-evaluation-plan-v1", "execution_contract_id": CONTRACT,
            "output": str(output), "release_path": str(release), "release_identity": identity(release),
            "actors": list(ACTORS), "primary_actor": "mirror50", "floor_seed": FLOOR_SEED,
            "controls_max": CONTROL_CAP, "normal_native_max": 5 * CONTROL_CAP, "compiler_native_max": 2,
            "original_six_controls": CASE_CONTROLS, "floor_controls": 1200,
            "export": {"ppo_load_max": 1, "torch_load_max": 3, "sb3_predictions": 384,
                       "numpy_predictions": 384, "mirror_predictions": 192, "tolerance": 1e-6,
                       "training_controls": 0, "physics_calls": 0, "environment_constructions": 0},
            "retry_permitted": False, "default_policy_replacement": False,
            "evaluation_scope": "previously observed six-task development set", **values}
    write(output / "plan.json", plan)
    return plan


def verified_plan50(output):
    require(output.is_absolute() and not any(p.is_symlink() for p in (output, *output.parents)),
            "C50 evaluation tree must be a regular absolute path")
    plan = read(output / "plan.json")
    require(plan["schema"] == "b22-sigma50-evaluation-plan-v1"
            and plan["execution_contract_id"] == CONTRACT and plan["output"] == str(output)
            and plan["actors"] == list(ACTORS) and plan["primary_actor"] == "mirror50"
            and plan["controls_max"] == CONTROL_CAP and plan["floor_controls"] == 1200
            and plan["normal_native_max"] == 5 * CONTROL_CAP and plan["compiler_native_max"] == 2
            and plan["original_six_controls"] == CASE_CONTROLS and plan["floor_seed"] == FLOOR_SEED
            and plan["export"] == {"ppo_load_max": 1, "torch_load_max": 3, "sb3_predictions": 384,
                                   "numpy_predictions": 384, "mirror_predictions": 192, "tolerance": 1e-6,
                                   "training_controls": 0, "physics_calls": 0, "environment_constructions": 0}
            and plan["retry_permitted"] is False, "C50 fixed evaluation plan differs")
    release = Path(plan["release_path"])
    require(identity(release) == plan["release_identity"], "C50 evaluation release changed")
    actual = inputs50(release, plan["training_stage"])
    require(all(plan[key] == value for key, value in actual.items()), "C50 evaluation final/source inputs changed")
    require(output.parent == parent().OUTPUT and output.name == "eval_" + plan["training_arm"],
            "C50 evaluation directory is not this model's registered location")
    return plan


def configure_paths50():
    environment = read(contract().MANIFEST)["runtime_environment"]
    sys.path[:0] = environment["PYTHONPATH"].split(os.pathsep)
    package = ModuleType("scripts")
    package.__path__ = [str(ROOT / "scripts")]
    sys.modules["scripts"] = package
    return environment


def derive50(kind, *, reading=False):
    """Change explicit identity nodes only; keep original arithmetic intact."""
    api = (training_reader().base_reader50() if reading else
           load("_sigma50_eval_worker_math", ROOT / "scripts/worker_b22_sigma50.py"))
    tree, previous = api.adapted_reader42(kind) if reading else api.adapt_module42(kind)
    original = copy.deepcopy(tree)
    edits = []

    def put(node, field, value, label):
        edits.append((node, field, getattr(node, field), label))
        setattr(node, field, value)

    def assignment(name, expected, replacement):
        found = [node for node in tree.body if isinstance(node, ast.Assign)
                 and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == name]
        require(len(found) == 1 and ast.dump(found[0].value, include_attributes=False)
                == ast.dump(ast.parse(expected, mode="eval").body, include_attributes=False),
                "C50 exact evaluation assignment changed: " + name)
        put(found[0], "value", ast.parse(replacement, mode="eval").body, "identity: " + name)

    if kind == "heldout":
        assignment("ACTORS", "('zero','control','deficit8')", repr(ACTORS))
        assignment("RECORD_SCHEMA", repr("d1-b22-tracking42-heldout-record-v1"), repr(RECORD_SCHEMA))
    if reading and kind == "score":
        for name, old in (("SCORE_SCHEMA", "case-score"), ("PAIR_SCHEMA", "pair-score"),
                          ("FLOOR_SCORE_SCHEMA", "floor-score")):
            assignment(name, repr("d1-b22-tracking42-" + old + "-v1"),
                       repr("d1-b22-sigma50-" + old + "-v1"))
        for name, old, new in (("score_case", ("zero", "control", "deficit8"), ACTORS),
                               ("score_pairs", ("control", "deficit8"), POLICIES),
                               ("score_floor_canonical", ("control", "deficit8"), POLICIES)):
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
            require(len(functions) == 1, "C50 scoring function missing")
            matches = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Tuple)
                       and all(isinstance(item, ast.Constant) for item in node.elts)
                       and tuple(item.value for item in node.elts) == old]
            require(len(matches) == 1, "C50 exact scoring actor enum changed")
            put(matches[0], "elts", [ast.Constant(item) for item in new], name + " actor enum")
    if reading and kind == "eval":
        matches = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                   and ast.unparse(node.targets[0]) == "(PAIR_FIELDS, RECORD_SCHEMA)"]
        require(len(matches) == 1 and isinstance(matches[0].value, ast.Tuple)
                and isinstance(matches[0].value.elts[1], ast.Constant)
                and matches[0].value.elts[1].value == "d1-b22-tracking42-heldout-record-v1",
                "C50 exact recorder schema import changed")
        put(matches[0].value.elts[1], "value", RECORD_SCHEMA, "independent reader record schema")
    derived = copy.deepcopy(tree)
    for node, field, old, _label in reversed(edits):
        setattr(node, field, old)
    require(ast.dump(tree, include_attributes=False) == ast.dump(original, include_attributes=False),
            "C50 evaluation arithmetic changed outside declared identity seams")
    return ast.fix_missing_locations(derived), {"base47": previous,
            "identity_seams": [row[3] for row in edits], "inverse_ast_exact": True,
            "derived_ast_sha256": hashlib.sha256(ast.dump(derived, include_attributes=False).encode()).hexdigest()}


def eval_math50(*, reading):
    worker = load("_sigma50_eval_recipe_worker", ROOT / "scripts/worker_b22_sigma50.py")
    spec = read(worker.SPEC_PATH)
    tree, recipe_proof = derive50("recipes")
    recipe = ModuleType("b22_recipes42")
    recipe.__file__ = str(ROOT / worker.SOURCES["recipes"][0])
    recipe._spec42 = spec
    sys.modules[recipe.__name__] = recipe
    exec(compile(tree, recipe.__file__, "exec"), recipe.__dict__)  # noqa: S102 -- inverse-checked frozen recipe
    proofs = {"recipes": recipe_proof}
    if not reading:
        tree, proofs["recorder"] = derive50("heldout")
        module = ModuleType("b22_sigma50_heldout")
        module.__file__ = str(ROOT / worker.SOURCES["heldout"][0])
        sys.modules[module.__name__] = module
        exec(compile(tree, module.__file__, "exec"), module.__dict__)  # noqa: S102 -- identity-only recorder derivation
        require(module.ACTORS == ACTORS and module.RECORD_SCHEMA == RECORD_SCHEMA, "C50 recorder identities differ")
        return SimpleNamespace(recipes=recipe, recorder=module, proofs=proofs)
    helper = training_reader().base_reader50()
    helper.assert_saved_only42()
    sys.meta_path.insert(0, helper._NoExecution42())
    helper.install_derived_verifier44()
    modules = {}
    for kind, name in (("score", "b22_score42"), ("eval", "b22_sigma50_eval_reader")):
        tree, proofs[kind] = derive50(kind, reading=True)
        module = ModuleType(name)
        module.__file__ = str(ROOT / helper.SOURCES[kind][0])
        module._spec42 = spec
        sys.modules[name] = module
        exec(compile(tree, module.__file__, "exec"), module.__dict__)  # noqa: S102 -- original independent saved-data arithmetic
        module.CONTRACT_ID = CONTRACT
        modules[kind] = module
    return SimpleNamespace(recipes=recipe, reader=modules["eval"], score=modules["score"], proofs=proofs)


def source_roots50(modules, sources):
    for module in modules:
        path = str(Path(module.__file__).resolve())
        require(path in sources and identity(Path(path)) == sources[path],
                "C50 loaded dependency root not bound to current closure: " + path)


def configured_numpy50(sources):
    """Configure before import and reject an already-loaded foreign NumPy."""
    configure_paths50()
    import numpy as np

    source_roots50((np,), sources)
    return np


def export50(output):
    plan = verified_plan50(output)
    folder = output / "export"
    folder.mkdir(exist_ok=False)
    report = {"status": "INCOMPLETE", "error": None, "plan_identity": identity(output / "plan.json"),
              "ppo_load_attempts": 0, "ppo_load_returns": 0, "torch_load_attempts": 0,
              "sb3_predict_attempts": 0, "sb3_predict_returns": 0,
              "numpy_predict_attempts": 0, "numpy_predict_returns": 0,
              "physics_calls": 0, "environment_constructions": 0, "training_calls": 0}
    original_load = None
    previous = {}
    started = time.monotonic()

    def interrupted(signum, _frame):
        raise InterruptedError(signal.Signals(signum).name)

    try:
        require("LD_PRELOAD" not in os.environ and "LD_LIBRARY_PATH" not in os.environ,
                "C50 export must have no physics preload")
        configure_paths50()
        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
        probe = load("_sigma50_export_pure_helpers", ROOT / "scripts/probe_b22_symmetry43.py")
        numeric = load("_sigma50_export_numpy", ROOT / "scripts/export_b22_numpy36.py")
        numeric.block_imports(numeric.SIMULATORS)
        # The historical pure probe defines helpers but also adjusts sys.path;
        # restore the registered path order before the first NumPy import.
        np = configured_numpy50(plan["source_hashes"])
        import stable_baselines3 as sb3
        import torch

        source_roots50((np, sb3, torch), plan["source_hashes"])
        checkpoint = Path(plan["training_run"]) / "final_checkpoint"
        metadata = read(checkpoint / "final_metadata.json")
        require((np.__version__, torch.__version__, sb3.__version__)
                == (metadata["numpy_version"], metadata["torch_version"], metadata["sb3_version"]),
                "C50 export dependency versions differ from training")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        original_load = torch.load

        def counted_load(*args, **kwargs):
            report["torch_load_attempts"] += 1
            require(report["torch_load_attempts"] <= 3, "C50 export Torch load cap")
            return original_load(*args, **kwargs)

        torch.load = counted_load
        report["ppo_load_attempts"] += 1
        model = sb3.PPO.load(checkpoint / "final_model.zip", device="cpu")
        report["ppo_load_returns"] += 1
        weights = probe.extract(model, np, torch)
        numeric.validate_weights(weights, np)
        symmetry = probe.load_symmetry()
        own = np.fromfile(checkpoint / "final_probe_observations_f32.bin", dtype="<f4").reshape(32, 99)
        historical = np.fromfile(FIXTURE, dtype="<f4").reshape(192, 99)[:160]
        cohort = np.concatenate([own, historical])
        reference, actual, symmetric = [], [], []

        def predict_numpy(obs):
            report["numpy_predict_attempts"] += 1
            require(report["numpy_predict_attempts"] <= 384, "C50 NumPy export call cap")
            action = numeric.numpy_action(obs, weights, np)[0]
            actual.append(action.copy())
            report["numpy_predict_returns"] += 1
            return action

        projected = symmetry.SymmetricActor(predict_numpy)
        for row in cohort:
            for observation in (row, symmetry.mirror_observation(row)):
                report["sb3_predict_attempts"] += 1
                require(report["sb3_predict_attempts"] <= 384, "C50 SB3 export call cap")
                action, state = model.predict(observation, deterministic=True)
                require(state is None and action.shape == (16,) and action.dtype == np.float32,
                        "C50 export actor result differs")
                reference.append(action.copy())
                report["sb3_predict_returns"] += 1
            symmetric.append(projected(row))
        reference, actual, symmetric = map(np.asarray, (reference, actual, symmetric))
        report["max_absolute_error"] = probe.compare(reference, actual, np)
        expected = np.fromfile(checkpoint / "final_probe_actions_f32.bin", dtype="<f4").reshape(32, 16)
        report["original_probe32_max_error"] = float(np.max(np.abs(reference[:64:2].astype(np.float64) - expected)))
        mirror_ref = np.asarray([(reference[i].astype(np.float64) + symmetry.mirror_action(reference[i + 1])) * .5
                                 for i in range(0, 384, 2)], dtype=np.float32)
        report["mirror_max_absolute_error"] = float(np.max(np.abs(symmetric.astype(np.float64) - mirror_ref)))
        require(report["original_probe32_max_error"] <= 1e-6 and report["mirror_max_absolute_error"] <= 1e-6
                and projected.actor_attempts == projected.actor_returns == 384
                and projected.predict_calls == 192 and report["torch_load_attempts"] == 3,
                "C50 complete export parity/call counts differ")
        report["mirror_predictions"] = projected.predict_calls
        np.savez(folder / "actor.npz", **weights)
        np.savez(folder / "numerical.npz", observations=cohort, sb3=reference, numpy=actual, mirror=symmetric)
        report["artifacts"] = {name: identity(folder / name) for name in ("actor.npz", "numerical.npz")}
        require(verified_plan50(output) == plan, "C50 export inputs changed")
        report.update(status="COMPLETE", checkpoint_sha256=plan["checkpoint_sha256"])
    except BaseException as exc:  # noqa: BLE001 -- retain partial/interrupted attempt
        report["error"] = f"{type(exc).__name__}: {exc}"
        write(folder / "error.json", {"traceback": traceback.format_exc()})
    finally:
        if original_load is not None:
            torch.load = original_load
        report["elapsed_s"] = time.monotonic() - started
        write(folder / "report.json", report)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return report


def checked_export50(output, plan):
    np = configured_numpy50(plan["source_hashes"])

    value = read(output / "export/report.json")
    require(value.get("status") == "COMPLETE" and value.get("error") is None
            and value.get("checkpoint_sha256") == plan["checkpoint_sha256"]
            and value.get("plan_identity") == identity(output / "plan.json")
            and value.get("ppo_load_attempts") == value.get("ppo_load_returns") == 1
            and value.get("torch_load_attempts") == 3
            and value.get("sb3_predict_attempts") == value.get("sb3_predict_returns") == 384
            and value.get("numpy_predict_attempts") == value.get("numpy_predict_returns") == 384
            and value.get("mirror_predictions") == 192
            and value.get("physics_calls") == value.get("environment_constructions") == value.get("training_calls") == 0
            and set(value.get("artifacts", {})) == {"actor.npz", "numerical.npz"},
            "C50 complete fixed-budget NumPy export required")
    for name, entry in value["artifacts"].items():
        require(identity(output / "export" / name) == entry, "C50 exported actor/numerical proof bytes changed")
    with np.load(output / "export/numerical.npz", allow_pickle=False) as saved:
        require(set(saved.files) == {"observations", "sb3", "numpy", "mirror"}, "C50 parity array member set differs")
        data = {name: saved[name] for name in saved.files}
    for name, shape in (("observations", (192, 99)), ("sb3", (384, 16)),
                        ("numpy", (384, 16)), ("mirror", (192, 16))):
        require(data[name].shape == shape and data[name].dtype == np.float32 and np.isfinite(data[name]).all(),
                "C50 parity array shape/type differs")
    checkpoint = Path(plan["training_run"]) / "final_checkpoint"
    observed = np.fromfile(checkpoint / "final_probe_observations_f32.bin", dtype="<f4").reshape(32, 99)
    historical = np.fromfile(FIXTURE, dtype="<f4").reshape(192, 99)[:160]
    require(np.array_equal(data["observations"], np.concatenate([observed, historical])),
            "C50 actual parity fixture differs")
    symmetry = load("_sigma50_saved_parity_symmetry", ROOT / "src/wheel_legged_control/d1/policy_symmetry43.py")
    mirrored = np.asarray([(data["sb3"][i].astype(np.float64) + symmetry.mirror_action(data["sb3"][i + 1])) * .5
                           for i in range(0, 384, 2)], dtype=np.float32)
    expected = np.fromfile(checkpoint / "final_probe_actions_f32.bin", dtype="<f4").reshape(32, 16)
    errors = {"max_absolute_error": float(np.max(np.abs(data["sb3"].astype(np.float64) - data["numpy"]))),
              "original_probe32_max_error": float(np.max(np.abs(data["sb3"][:64:2].astype(np.float64) - expected))),
              "mirror_max_absolute_error": float(np.max(np.abs(data["mirror"].astype(np.float64) - mirrored)))}
    require(all(math.isfinite(number) and number <= 1e-6 and value.get(name) == number
                for name, number in errors.items()), "C50 independent saved parity differs")
    return value


def worker50(output):
    plan = verified_plan50(output)
    exported = checked_export50(output, plan)
    folder = output / "physics"
    folder.mkdir(exist_ok=False)
    write(folder / "session.json", {"schema": "b22-sigma50-eval-session-v1",
          "execution_contract_id": CONTRACT, "plan_identity": identity(output / "plan.json"),
          "source_hashes": plan["source_hashes"], "parent_pid": os.getppid(),
          "export_report_identity": identity(output / "export/report.json"),
          "checkpoint_sha256": plan["checkpoint_sha256"], "retry_permitted": False})
    result = {"status": "INCOMPLETE", "error": None, "cases": [], "cleanup_errors": [],
              "plan_identity": identity(output / "plan.json"), "counts": None, "known_counts": None,
              "training": 0, "ppo_load": 0, "torch_load": 0, "numpy_bundle_loads": 0}
    runtime = guard = env = servo = writer = quota = backend = None
    actors = {}
    old_warning = None
    warning_installed = False
    previous = {}
    started = time.monotonic()
    try:
        environment = configure_paths50()
        library = environment["LD_PRELOAD"]
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and len(os.sched_getaffinity(0)) == 1,
                "C50 evaluation requires Nice10 and one CPU")
        require(os.environ.get("LD_PRELOAD") == library and os.environ.get("LD_BIND_NOW") == "1"
                and "LD_LIBRARY_PATH" not in os.environ, "C50 counted engine preload required")
        backend = load("_sigma50_forbidden_frameworks", ROOT / "scripts/b22_numpy_backend38.py")
        backend.deny_frameworks38()
        # isort: off
        import mujoco
        import mujoco._functions
        import mujoco._rollout
        import engine_binding
        import numpy as np
        from archive13.atomic_archive_13 import ArchiveWriter, AtomicCourseNativeGuard
        from controller18 import install_controller18
        from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
        from run_rl16_training_08 import _boundary, _control_step_caller_verified
        from run_world_upright_reference_11 import _binding
        from world_upright_course_11 import WorldUprightCourseEnv
        from scripts import run_d1_latest_rl as common
        from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
        from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
        from wheel_legged_control.d1.yaw_feedforward44 import adopt_yaw_feedforward
        # isort: on

        source_roots50((np, mujoco, engine_binding), plan["source_hashes"])
        require(_nominal_geometry.cache_info().misses == 0, "C50 cold geometry required")
        worker_api = load("_sigma50_eval_worker_counting", ROOT / "scripts/worker_b22_sigma50.py")
        quota = worker_api.StageQuota42(folder, 2 << 30, execution_output=output, global_stop=3 << 30)
        # The original before-write quota includes compressed/partial payloads.
        writer = worker_api.quota_writer42(ArchiveWriter, quota, common._jsonable)()
        old_warning = mujoco.get_mju_user_warning()

        def warning(message):
            raise RuntimeError("MuJoCo warning: " + str(message))

        mujoco.set_mju_user_warning(warning)
        warning_installed = True

        def stop(*_args):
            writer.request_stop()

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}

        class Runtime(RollingEngineRuntime):
            def control_step(self, current_env, action):
                require(not writer.stop_requested, "C50 evaluation stop requested")
                return super().control_step(current_env, action)

        with np.load(output / "export/actor.npz", allow_pickle=False) as saved:
            weights = {key: saved[key] for key in saved.files}
        result["numpy_bundle_loads"] = 1
        require(identity(output / "export/actor.npz") == exported["artifacts"]["actor.npz"],
                "C50 actual NumPy actor bytes changed")
        numeric = load("_sigma50_eval_numpy_shape", ROOT / "scripts/export_b22_numpy36.py")
        numeric.validate_weights(weights, np)
        actor_helper = load("_sigma50_eval_actor_math", ROOT / "scripts/b22_symmetry_physics43.py")
        actors = {name: actor_helper.Actor(weights, symmetric=name == "mirror50",
                                          limit=10400 if name == "raw50" else 20800) for name in POLICIES}
        maths = eval_math50(reading=False)
        worker_api.install_servo_counter42({"pure_servo_advance_max": CONTROL_CAP})
        servo = worker_api._servo_counter42
        with (Runtime(Path(library), control_limit=CONTROL_CAP, construction_limit=2) as runtime,
              AtomicCourseNativeGuard(runtime, writer) as guard):
            writer.commit_json(folder / "derivation_worker.json", maths.proofs)
            env = runtime.construct(lambda: WorldUprightCourseEnv(
                caps=QualifiedCommandCaps(1.6, 0., 0., .3, False),
                command_source=lambda *_: FullDriveCommand(),
                spawn_position_m=(-8., -4.7, .455), max_steps=1000, mode="eval"))
            controller = install_controller18(env, "combined")
            adopt_yaw_feedforward(controller._nominal, 3.9)
            contract().keep_record_consistent(controller, 3.9)
            state = runtime.state()
            require(state["construction_attempts"] == state["construction_returns"] == 2
                    and _nominal_geometry.cache_info().misses == 1, "C50 cold construction ledger differs")
            writer.commit_json(folder / "construction_receipt.json", {
                "C_state": state, "nominal_cache": _nominal_geometry.cache_info()._asdict(),
                "compiled_geometry": env.plant.collision_terrain_metadata,
                "geometry_binding": _binding(env.plant), "proof": runtime.proof, "boundary": _boundary(runtime)})
            env.set_native_interval_reader(guard.interval_summary, begin_interval=guard.begin_interval)
            (folder / "floor").mkdir()
            floor_initial = None
            for actor in POLICIES:
                receipt, initial = maths.recorder.record_case(
                    runtime, guard, env, actors[actor], maths.recipes.floor_schedule(FLOOR_SEED), actor,
                    folder / "floor" / actor, writer=writer, pair_initial=floor_initial,
                    checkpoint_sha256=plan["checkpoint_sha256"])
                require(receipt["record_valid"], "C50 floor record invalid")
                if floor_initial is None:
                    floor_initial = initial
                result["cases"].append({"folder": "floor/" + actor, "controls": receipt["completed_controls"]})
            (folder / "heldout").mkdir()
            for case in maths.recipes.ORDER:
                paired = None
                for actor in ACTORS:
                    receipt, initial = maths.recorder.record_case(
                        runtime, guard, env, actors.get(actor), maths.recipes.make_schedule(case), actor,
                        folder / "heldout" / (case + "_" + actor), writer=writer, pair_initial=paired,
                        checkpoint_sha256=plan["checkpoint_sha256"] if actor != "zero" else None)
                    require(receipt["record_valid"], "C50 task record invalid")
                    if paired is None:
                        paired = initial
                    result["cases"].append({"folder": "heldout/" + case + "_" + actor,
                                            "controls": receipt["completed_controls"]})
            state = runtime.state()
            caller = _control_step_caller_verified(state, runtime.proof, plan["source_hashes"], engine_binding)
            require(runtime.control_attempted == runtime.control_completed
                    and runtime.attempted == runtime.returned == runtime.advanced_substeps == 5 * runtime.control_completed
                    and runtime.failed == 0 and guard.attempted == guard.returned == guard.checked == runtime.returned
                    and guard.failure is None and state["control_attempts"] == state["control_returns"] == runtime.returned
                    and state["native_construction_caller_verified"] is True and caller
                    and state["violations"] == 0 and runtime.ledger_receipt()["forbidden_entries"] == 0
                    and state["ccd_attempts"] == state["ccd_returns"]
                    and (state["ccd_attempts"] == 0 or state["native_ccd_caller_verified"] is True),
                    "C50 native count/caller mismatch")
            result.update(status="COMPLETE", native_caller_verified=True,
                          counts={"controls": runtime.control_completed, "normal_native": runtime.returned,
                                  "compiler_native": state["construction_returns"]})
            result["native_guard"] = guard.report()
        require(verified_plan50(output) == plan, "C50 evaluation source closure changed")
    except BaseException as exc:  # noqa: BLE001 -- preserve incomplete native/archive evidence
        result["status"] = "INCOMPLETE"
        result["error"] = f"{type(exc).__name__}: {exc}"
        write(folder / "worker_error.json", {"traceback": traceback.format_exc()})
    finally:
        for target in (env, servo):
            if target is not None:
                try:
                    target.close()
                except BaseException as exc:  # noqa: BLE001 -- retain cleanup failures
                    result["cleanup_errors"].append(repr(exc))
        if warning_installed:
            mujoco.set_mju_user_warning(old_warning)
        result["actor_counts"] = {key: value.counts() for key, value in actors.items()}
        if runtime is not None:
            result["C_final"] = runtime.state()
            result["python"] = runtime.ledger_receipt()
            result["known_counts"] = {"controls": runtime.control_completed, "normal_native": runtime.returned,
                                      "compiler_native": runtime.state()["construction_returns"]}
        result["pure_servo"] = None if servo is None else servo.report()
        result["quota"] = None if quota is None else quota.report()
        result["forbidden_frameworks_absent"] = None if backend is None else backend.frameworks_absent38()
        result["elapsed_s"] = time.monotonic() - started
        if (writer is not None and (writer.failed or writer.stop_requested)) or result["cleanup_errors"]:
            result["status"] = "INCOMPLETE"
        if result["status"] != "COMPLETE":
            result["counts"] = None
        write(folder / "worker_report.json", result)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return result


def read50(output):
    plan = verified_plan50(output)
    checked_export50(output, plan)
    folder = output / "physics"
    attempt = output / "read_attempt.json"
    write(attempt, {"schema": "b22-sigma50-eval-reader-attempt-v1",
                   "plan_identity": identity(output / "plan.json"), "retry_permitted": False})
    result = {"status": "INCOMPLETE", "error": None, "plan_identity": identity(output / "plan.json"),
              "execution_contract_id": CONTRACT, "reader_model_calls": 0, "reader_physics_calls": 0,
              "qualification_granted": False, "default_policy_replacement": False}
    try:
        configure_paths50()
        helper = training_reader().base_reader50()
        helper.assert_saved_only42()
        maths = eval_math50(reading=True)
        reader = maths.reader
        worker = read(folder / "worker_report.json")
        session = read(folder / "session.json")
        require(worker["status"] == "COMPLETE" and worker["error"] is None and not worker["cleanup_errors"]
                and worker["plan_identity"] == identity(output / "plan.json")
                and session["execution_contract_id"] == CONTRACT
                and session["plan_identity"] == identity(output / "plan.json")
                and session["source_hashes"] == plan["source_hashes"]
                and session["checkpoint_sha256"] == plan["checkpoint_sha256"]
                and session["export_report_identity"] == identity(output / "export/report.json"),
                "C50 current complete eval worker/session required")
        construction = reader.document(folder / "construction_receipt.json")
        binding = reader.check_binding(construction["geometry_binding"])
        geometry = construction["compiled_geometry"]
        require(len(geometry["world_collision_geoms"]) == 92 and geometry["control_dt_s"] == .01
                and geometry["native_dt_s"] == .002, "C50 compiled geometry dimensions differ")
        require(len(maths.recipes.ORDER) == 6 and set(maths.recipes.ORDER) == set(maths.score.ORIGINAL_SIX),
                "C50 exact original six tasks required")
        scores, summaries, floors, parsed, floor_initial, floor_reset = [], [], {}, [], [], []
        with helper.geometry_counter42(sys.modules["geometry21"], CONTROL_CAP) as pure:
            floor_spec = ("floor_0p4_600", "flat", .4, FLOOR_SEED, 600, 325, 425)
            for actor in POLICIES:
                summary, canonical = reader.heldout_case(folder, floor_spec, actor, binding, geometry,
                    expected_variant="combined", folder_override=folder / "floor" / actor)
                require(canonical["checkpoint_sha256"] == plan["checkpoint_sha256"], "C50 floor model differs")
                floors[actor] = maths.score.score_floor_canonical(canonical, expected_seed=FLOOR_SEED)
                summaries.append(summary)
                floor_initial.append(reader.arrays(folder / "floor" / actor / "initial_state.npz"))
                floor_reset.append(reader._reset_state(reader.document(folder / "floor" / actor / "reset_receipt.json")))
            require(reader._same_arrays(*floor_initial) and helper.same_json42(*floor_reset),
                    "C50 floor initial/controller states differ")
            for case in maths.recipes.ORDER:
                task = maths.recipes.TASK_BY_ID[case]
                spec = (case, task.terrain, task.speed_mps, task.seed, task.horizon, task.hold_start, task.release_tick)
                parsed.append((spec, False, ACTORS))
                for actor in ACTORS:
                    summary, canonical = reader.heldout_case(folder, spec, actor, binding, geometry,
                                                             expected_variant="combined")
                    require(canonical["checkpoint_sha256"] == (plan["checkpoint_sha256"] if actor != "zero" else None),
                            "C50 actual case checkpoint differs")
                    score = maths.score.score_case(canonical, expected_seed=task.seed, mirror=False)
                    scores.append(dict(score, experiment_actor=actor, checkpoint_sha256=canonical["checkpoint_sha256"]))
                    summaries.append(summary)
        pairs = reader._new_pairs(folder, parsed, list(POLICIES))
        require({p.name for p in (folder / "heldout").iterdir() if p.is_dir()}
                == {case + "_" + actor for case in maths.recipes.ORDER for actor in ACTORS}
                and {p.name for p in (folder / "floor").iterdir() if p.is_dir()} == set(POLICIES),
                "C50 omitted or unexpected evaluation case")
        total = sum(row["completed_controls"] for row in summaries)
        require(total == worker["counts"]["controls"] <= CONTROL_CAP
                and worker["counts"]["normal_native"] == 5 * total and worker["counts"]["compiler_native"] == 2
                and pure["attempted"] == pure["returned"]
                == sum(row["geometry21"]["pure_servo_intervals"] for row in summaries),
                "C50 independent observed physical/geometry counts differ")
        c, py, ng = worker["C_final"], worker["python"], worker["native_guard"]
        require(c["construction_attempts"] == c["construction_returns"] == 2
                and c["control_attempts"] == c["control_returns"] == 5 * total
                and c["phase"] == c["target_model"] == c["target_data"] == c["violations"] == 0
                and py["control_attempted"] == py["control_completed"] == total
                and py["native_attempted"] == py["native_returned"] == py["clock_advanced_substeps"] == 5 * total
                and py["native_failed"] == py["forbidden_entries"] == 0
                and ng["native_attempted"] == ng["native_returned"] == ng["native_checked"] == 5 * total
                and ng["failure"] is None and worker["forbidden_frameworks_absent"] is True
                and worker["quota"]["strict_worker_before_write"] is True
                and worker["quota"]["exhausted"] is False and worker["quota"]["limit_bytes"] == 2 << 30,
                "C50 native/Python/guard/quota closure differs")
        for actor, multiplier in (("raw50", 1), ("mirror50", 2)):
            predictions = sum(row["predict_calls"] for row in summaries if row["actor"] == actor)
            require(worker["actor_counts"][actor] == {"predictions": predictions, "attempts": multiplier * predictions,
                                                      "returns": multiplier * predictions},
                    "C50 actual NumPy inference counts differ")
        require(worker["training"] == worker["ppo_load"] == worker["torch_load"] == 0
                and worker["numpy_bundle_loads"] == 1, "C50 evaluation used a forbidden model/training path")
        summary = descriptive50(scores, maths.recipes.ORDER)
        result.update(status="COMPLETE", training_arm=plan["training_arm"], training_seed=plan["training_seed"],
                      training_coverage_valid=plan["training_coverage_valid"], training_coverage=plan["training_coverage"],
                      checkpoint_sha256=plan["checkpoint_sha256"], cases=scores, floors=floors, pairs=pairs,
                      summaries=summaries, derivation=maths.proofs,
                      contribution={actor: maths.score.score_pairs(scores, policy_actor=actor) for actor in POLICIES},
                      diagnostic=summary, counts=worker["counts"], pure_servo=pure,
                      floors_screened_models=False, worker_report_identity=identity(folder / "worker_report.json"),
                      artifacts=helper.artifact_inputs42(folder))
        require(verified_plan50(output) == plan, "C50 saved reader sources changed")
        helper.assert_saved_only42()
    except BaseException as exc:  # noqa: BLE001 -- preserve independently detected failure
        result.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
        write(output / "read_error.json", {"traceback": traceback.format_exc()})
    write(output / "readback.json", result)
    return result


def descriptive50(scores, cases):
    """Full fixed drive-window R; never turn a partial trajectory into n=3."""
    require(len(cases) == 6 and len(set(cases)) == 6 and len(scores) == 18,
            "C50 complete 18-entry task matrix required")
    indexed = {(row["case_id"], row["experiment_actor"]): row for row in scores}
    require(set(indexed) == {(case, actor) for case in cases for actor in ACTORS},
            "C50 task matrix missing/duplicated/foreign actor")
    complete = {actor: all(indexed[case, actor]["completed_controls"]
                           == indexed[case, actor]["windows"]["control_cap"] for case in cases) for actor in ACTORS}
    pooled = {actor: sum(indexed[case, actor]["rl_terms"]["sse_total"] for case in cases) for actor in ACTORS}
    require(all(type(value) in (float, int) and math.isfinite(value) and value >= 0 for value in pooled.values()),
            "C50 finite nonnegative pooled SSE required")
    defined = complete["zero"] and complete["mirror50"] and pooled["zero"] > 0
    return {"primary_actor": "mirror50", "full_horizon_by_actor": complete,
            "all_original_cases_full_horizon": all(complete.values()),
            "pooled_drive_sse": pooled,
            "R": {actor: pooled[actor] / pooled["zero"] if complete["zero"] and complete[actor] and pooled["zero"] > 0
                   else None for actor in POLICIES}, "primary_defined": defined,
            "reason": None if defined else "incomplete task horizon or nonpositive zero denominator",
            "scope": "fixed six-task development diagnostic; no independent generalization claim"}


def compare_diagnostics50(records, arms):
    """Three predeclared pairs; failures remain visible, never imputed."""
    require(set(records) <= set(arms), "C50 unregistered evaluation must not be silently dropped")
    rows = []
    for arm, spec in arms.items():
        value = records.get(arm)
        status = "NOT_RUN" if value is None else value.get("status", "INCOMPLETE")
        diagnostic = {} if value is None else value.get("diagnostic", {})
        valid = (status == "COMPLETE" and value.get("training_arm") == arm
                 and value.get("training_seed") == spec["seed"]
                 and diagnostic.get("primary_actor") == "mirror50"
                 and diagnostic.get("primary_defined") is True)
        ratio = diagnostic.get("R", {}).get("mirror50") if valid else None
        valid = valid and type(ratio) in (int, float) and math.isfinite(ratio) and ratio >= 0
        rows.append({"arm": arm, "seed": spec["seed"], "sigma_multiplier": spec["sigma_multiplier"],
                     "status": status, "primary_defined": bool(valid), "R": ratio if valid else None,
                     "training_coverage_valid": None if value is None else value.get("training_coverage_valid"),
                     "training_coverage": None if value is None else value.get("training_coverage"),
                     "contribution": None if value is None else value.get("contribution"),
                     "error": None if value is None else value.get("error")})
    complete = len(rows) == 6 and all(row["primary_defined"] for row in rows)
    result = {"schema": "b22-sigma50-three-pair-development-diagnostic-v1", "status": "COMPLETE" if complete else "INCOMPLETE",
              "primary_actor": "mirror50", "rows": rows, "n_registered_pairs": 3,
              "n_complete_pairs": sum(all(any(row["seed"] == seed and row["sigma_multiplier"] == arm
                                                 and row["primary_defined"] for row in rows) for arm in (1., .5))
                                      for seed in (201051, 201061, 201071)),
              "direction_pass": False if complete else None, "mean": None, "range": None,
              "statistical_significance_claimed": False, "default_policy_replacement": False,
              "qualification_granted": False,
              "scope": "n=3 descriptive PPO-seed range conditional on fixed environment streams and development tasks"}
    if complete:
        values = {name: [row["R"] for row in rows if row["sigma_multiplier"] == multiplier]
                  for name, multiplier in (("baseline", 1.), ("candidate", .5))}
        require(all(len(value) == 3 for value in values.values()), "C50 exactly three pairs required")
        mean = {name: sum(value) / 3 for name, value in values.items()}
        span = {name: max(value) - min(value) for name, value in values.items()}
        result.update(mean=mean, range=span,
                      direction_pass=span["candidate"] < span["baseline"] and mean["candidate"] <= mean["baseline"],
                      paired_difference=[{"seed": seed, "candidate_minus_baseline":
                          next(row["R"] for row in rows if row["seed"] == seed and row["sigma_multiplier"] == .5)
                          - next(row["R"] for row in rows if row["seed"] == seed and row["sigma_multiplier"] == 1.)}
                          for seed in (201051, 201061, 201071)])
    return result


def compare50(output):
    require(output == parent().OUTPUT, "C50 comparison uses the registered run only")
    records, provenance = {}, {}
    for arm in contract().ARMS:
        folder = output / ("eval_" + arm)
        path = folder / "readback.json"
        if not path.exists():
            if any((folder / name).exists() for name in ("export", "physics", "read_attempt.json")):
                records[arm] = {"status": "INCOMPLETE", "error": "evaluation attempt lacks completed independent readback"}
            continue
        value = read(path)
        if value.get("status") == "COMPLETE":
            plan = verified_plan50(folder)
            require(value["plan_identity"] == identity(folder / "plan.json")
                    and value["execution_contract_id"] == CONTRACT
                    and value["checkpoint_sha256"] == plan["checkpoint_sha256"],
                    "C50 comparison readback source differs")
            for name, entry in value["artifacts"].items():
                relative = Path(name)
                require(not relative.is_absolute() and ".." not in relative.parts
                        and identity(folder / "physics" / relative) == entry, "C50 post-read evaluation bytes changed")
        records[arm] = value
        provenance[arm] = identity(path)
    result = compare_diagnostics50(records, contract().ARMS)
    result["readback_identities"] = provenance
    write(output / "training_diagnostic50.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "export", "worker", "read", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--stage")
    args = parser.parse_args(argv)
    require(args.output.is_absolute(), "C50 absolute output required")
    if args.mode == "prepare":
        require(args.release is not None and args.stage is not None, "C50 release and completed training stage required")
        result = prepare50(args.release, args.stage, args.output)
        print(json.dumps({"status": "PREPARED", "controls_max": result["controls_max"]}))
        return 0
    require(args.release is None and args.stage is None, "C50 execution uses only its sealed plan")
    result = {"export": export50, "worker": worker50, "read": read50, "compare": compare50}[args.mode](args.output)
    print(json.dumps({"status": result["status"], "mode": args.mode}))
    return int(result["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
