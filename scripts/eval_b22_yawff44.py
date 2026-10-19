"""Evaluate the completed 42 deficit8 actor (raw and mirror-averaged) on the yaw-feedforward baseline.

Same six original B22 tasks, same NumPy actors and same independent reader chain as the 43 run;
the only difference is that the nominal controller adopts ``yaw_command_feedforward`` before the
first reset, for every actor including zero. It answers: on the improved baseline, how much of the
remaining tracking error does the existing residual actor still remove (the original gate is
pooled SSE <= 0.85 x zero)? No training, no new weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).resolve()
RESULTS = ROOT / "results"
MANIFEST = ROOT / "runtime/d1_b22/runtime_manifest_v27.json"
NUMERIC = ROOT / "results/b22_symmetry43_numeric_20261018_03"
MODEL_SHA = "44cbf9a4363ff42af78d8e54049e0dbade2bea74701e9a27073bcb9e656f3239"
ACTORS = ("zero", "raw42", "mirror43")
WALL_S = 2400
CASES: tuple[str, ...] = ()
OUTPUT = RESULTS / "b22_yawff44_actors_unset"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def configure_paths():
    sys.path[:0] = read(MANIFEST)["runtime_environment"]["PYTHONPATH"].split(os.pathsep)
    from types import ModuleType

    package = ModuleType("scripts")
    package.__path__ = [str(ROOT / "scripts")]
    sys.modules["scripts"] = package


def folder_of(feedforward: float) -> Path:
    return OUTPUT / f"FF{feedforward:g}"


def worker(feedforward: float) -> int:
    folder = folder_of(feedforward)
    folder.mkdir(parents=True, exist_ok=True)
    h = load("yawff44_helpers", ROOT / "scripts/b22_symmetry_physics43.py")
    configure_paths()
    backend = load("yawff44_forbidden", ROOT / "scripts/b22_numpy_backend38.py")
    backend.deny_frameworks38()
    # isort: off
    import mujoco
    import mujoco._functions
    import mujoco._rollout
    import engine_binding  # noqa: F401
    import numpy as np
    from archive13.atomic_archive_13 import ArchiveWriter, AtomicCourseNativeGuard
    from controller18 import install_controller18
    from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
    from run_rl16_training_08 import _boundary
    from run_world_upright_reference_11 import _binding
    from world_upright_course_11 import WorldUprightCourseEnv

    from scripts import run_d1_latest_rl as common
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    from wheel_legged_control.d1.yaw_feedforward44 import adopt_yaw_feedforward
    # isort: on

    explore = load("yawff44_explore", ROOT / "scripts/explore_b22_yawgain44.py")
    package = h.api42().package42()
    package.verify_loaded_roots39(package.inventory39())

    class Writer(ArchiveWriter):
        def commit_json(self, path, value):
            return super().commit_json(path, common._jsonable(value))

    writer = Writer()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, lambda *_: writer.request_stop())
    signal.setitimer(signal.ITIMER_REAL, WALL_S - 20)
    old_warning = mujoco.get_mju_user_warning()

    def warning(message):
        raise RuntimeError("MuJoCo warning: " + str(message))

    mujoco.set_mju_user_warning(warning)
    result = {"status": "INCOMPLETE", "error": None, "feedforward": feedforward, "cases": []}
    env = servo = runtime = None
    actors = {}
    try:
        with np.load(NUMERIC / "actor.npz", allow_pickle=False) as saved:
            weights = {k: saved[k] for k in saved.files}
        math36 = load("yawff44_numpy_shape", ROOT / "scripts/export_b22_numpy36.py")
        math36.validate_weights(weights, np)
        recipe, recorder, _ = h.recorder()
        order = CASES or recipe.ORDER
        # one NumPy forward per control tick (mirror: two); generous but bounded
        horizon = 1800 * len(order)
        actors = {
            name: h.Actor(weights, symmetric=name == "mirror43", limit=2 * horizon + 400)
            for name in ("raw42", "mirror43")
        }
        controls_max = horizon * len(ACTORS)
        api = h.api42()
        api.install_servo_counter42({"pure_servo_advance_max": controls_max})
        servo = api._servo_counter42
        library = read(MANIFEST)["runtime_environment"]["LD_PRELOAD"]
        with (
            RollingEngineRuntime(
                Path(library), control_limit=controls_max, construction_limit=2
            ) as runtime,
            AtomicCourseNativeGuard(runtime, writer) as guard,
        ):
            env = runtime.construct(
                lambda: WorldUprightCourseEnv(
                    caps=QualifiedCommandCaps(1.6, 0.0, 0.0, 0.3, False),
                    command_source=lambda *_: FullDriveCommand(),
                    spawn_position_m=(-8.0, -4.7, 0.455),
                    max_steps=1000,
                    mode="eval",
                )
            )
            controller = install_controller18(env, "combined")
            adopt_yaw_feedforward(controller._nominal, feedforward)
            explore.keep_record_consistent(controller, feedforward)
            writer.commit_json(
                folder / "construction_receipt.json",
                {
                    "C_state": runtime.state(),
                    "nominal_cache": _nominal_geometry.cache_info()._asdict(),
                    "compiled_geometry": env.plant.collision_terrain_metadata,
                    "geometry_binding": _binding(env.plant),
                    "proof": runtime.proof,
                    "boundary": _boundary(runtime),
                },
            )
            env.set_native_interval_reader(
                guard.interval_summary, begin_interval=guard.begin_interval
            )
            (folder / "heldout").mkdir()
            for case in order:
                paired = None
                for actor in ACTORS:
                    receipt, initial = recorder.record_case(
                        runtime,
                        guard,
                        env,
                        actors.get(actor),
                        recipe.make_schedule(case),
                        actor,
                        folder / "heldout" / f"{case}_{actor}",
                        writer=writer,
                        pair_initial=paired,
                        checkpoint_sha256=MODEL_SHA if actor != "zero" else None,
                    )
                    if not receipt["record_valid"]:
                        raise RuntimeError(f"invalid record {case}/{actor}")
                    if paired is None:
                        paired = initial
                    result["cases"].append(
                        {"case": case, "actor": actor, "controls": receipt["completed_controls"]}
                    )
                    (folder / "progress.json").write_text(json.dumps(result))
            result["status"] = "COMPLETE"
            result["controls"] = runtime.control_completed
    except BaseException as exc:  # noqa: BLE001 -- keep the failure with the partial evidence
        result["error"] = f"{type(exc).__name__}: {exc}"
        (folder / "worker_error.txt").write_text(traceback.format_exc())
    finally:
        for target in (env, servo):
            if target is not None:
                try:
                    target.close()
                except BaseException as exc:  # noqa: BLE001
                    result.setdefault("cleanup_errors", []).append(repr(exc))
        mujoco.set_mju_user_warning(old_warning)
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["actor_counts"] = {k: v.counts() for k, v in actors.items()}
        if writer.failed or writer.stop_requested or result.get("cleanup_errors"):
            result["status"] = "INCOMPLETE"
        (folder / "worker_report.json").write_text(json.dumps(result, indent=2))
    return int(result["status"] != "COMPLETE")


def read_scores(feedforward: float) -> int:
    """Independent saved-data scoring through the 43 reader chain; no model, no physics."""
    folder = folder_of(feedforward)
    h = load("yawff44_helpers", ROOT / "scripts/b22_symmetry_physics43.py")
    configure_paths()
    explore = load("yawff44_explore", ROOT / "scripts/explore_b22_yawgain44.py")
    session = folder / "session.json"
    if not session.exists():
        reference = read(ROOT / "results/b22_symmetry43_physics_20261018_01/session.json")
        session.write_text(
            json.dumps(
                {
                    "execution_contract_id": reference["execution_contract_id"],
                    "source_hashes": reference["source_hashes"],
                    "provenance": "copied from the completed 43 session; evaluation, not a qualification",
                }
            )
        )
    explore.derive_verifier(feedforward)
    maths = h.saved_math()
    reader = maths.reader
    worker_result = read(folder / "worker_report.json")
    if worker_result["status"] != "COMPLETE" or worker_result["error"] is not None:
        raise RuntimeError("worker incomplete")
    construction = reader.document(folder / "construction_receipt.json")
    binding = reader.check_binding(construction["geometry_binding"])
    geometry = construction["compiled_geometry"]
    pure_helper = h.reader42()
    scores, parsed = [], []
    order = CASES or maths.recipes.ORDER
    with pure_helper.geometry_counter42(sys.modules["geometry21"], 1800 * len(order) * len(ACTORS)):
        for case in order:
            task = maths.recipes.TASK_BY_ID[case]
            spec = (
                case,
                task.terrain,
                task.speed_mps,
                task.seed,
                task.horizon,
                task.hold_start,
                task.release_tick,
            )
            mirror = bool(maths.recipes.CASE_SPECS[case].get("mirror", False))
            parsed.append((spec, mirror, ACTORS))
            for actor in ACTORS:
                _, canonical = reader.heldout_case(
                    folder,
                    spec,
                    actor,
                    binding,
                    geometry,
                    expected_variant="combined",
                    mirror=mirror,
                )
                if canonical["checkpoint_sha256"] != (MODEL_SHA if actor != "zero" else None):
                    raise RuntimeError("checkpoint identity differs")
                score = maths.score.score_case(canonical, expected_seed=task.seed, mirror=mirror)
                scores.append(
                    dict(
                        score,
                        experiment_actor=actor,
                        checkpoint_sha256=canonical["checkpoint_sha256"],
                    )
                )
    pairs = reader._new_pairs(folder, parsed, ["raw42", "mirror43"])
    result = {
        "feedforward": feedforward,
        "cases": scores,
        "pairs": pairs,
        "contribution": {
            actor: maths.score.score_pairs(scores, policy_actor=actor)
            for actor in ("raw42", "mirror43")
        },
        "actor_counts": worker_result["actor_counts"],
    }
    (folder / "readback.json").write_text(json.dumps(result, default=str))
    return 0


def execute(feedforwards: tuple[float, ...]) -> int:
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    codes = {}
    for feedforward in feedforwards:
        env = os.environ.copy()
        for key, value in read(MANIFEST)["runtime_environment"].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        extra = ["--output", OUTPUT.name] + (["--cases", ",".join(CASES)] if CASES else [])
        with (OUTPUT / f"worker_FF{feedforward:g}.log").open("x") as log:
            child = subprocess.Popen(
                [
                    "/usr/bin/python3",
                    str(DRIVER),
                    "--worker",
                    "--feedforward",
                    str(feedforward),
                    *extra,
                ],
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                child.wait(timeout=WALL_S)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=20)
        codes[f"worker_{feedforward:g}"] = child.returncode
        read_env = {k: v for k, v in env.items() if k != "LD_PRELOAD"}
        with (OUTPUT / f"read_FF{feedforward:g}.log").open("x") as log:
            reader_child = subprocess.run(
                [
                    "/usr/bin/python3",
                    str(DRIVER),
                    "--read",
                    "--feedforward",
                    str(feedforward),
                    *extra,
                ],
                env=read_env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=WALL_S,
                check=False,
            )
        codes[f"read_{feedforward:g}"] = reader_child.returncode
    print(json.dumps({"returncodes": codes, "elapsed_s": time.monotonic() - started}))
    return int(any(codes.values()))


def main() -> int:
    global OUTPUT, CASES
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    group.add_argument("--read", action="store_true")
    parser.add_argument("--feedforward", type=float)
    parser.add_argument("--feedforwards", default="3.9")
    parser.add_argument("--output", required=True, help="directory name under results/")
    parser.add_argument(
        "--cases", help="comma list of preregistered case ids; default six originals"
    )
    args = parser.parse_args()
    OUTPUT = RESULTS / args.output
    if args.cases:
        CASES = tuple(args.cases.split(","))
    if args.read:
        return read_scores(args.feedforward)
    if args.worker:
        return worker(args.feedforward)
    return execute(tuple(float(x) for x in args.feedforwards.split(",")))


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(main())
