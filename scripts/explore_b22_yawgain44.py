"""Exploratory zero-residual yaw-law study on the six original B22 tasks.

NOT a qualification run: no plan/readback chain, only a descriptive scorer. It tests the
hypothesis from the saved-data diagnostic that the zero-residual baseline under-delivers yaw
rate (closed-loop gain ~0.63 at feedback gain 4). Output _02 raised the feedback gain to 16
(`controller._nominal.yaw_feedback_gain = 16`): the turning task improved but every
non-turning task regressed, so this version adds only a command feedforward term
`effective_yaw = ff*command + 4*(command - measured)`. Each setting runs in its own worker
with the same initial state per task; no actor, no training, no residual.
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
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).resolve()
OUTPUT = ROOT / "results/b22_yawgain44_explore_20261018_05"  # replaced by --output
CASES: tuple[str, ...] = ()  # empty means the six original tasks; replaced by --cases
MANIFEST = ROOT / "runtime/d1_b22/runtime_manifest_v27.json"
FEEDFORWARDS = (3.9,)  # feedback gain stays at the frozen 4.0
WALL_S = 900
CONTROLS_MAX = 1800 * 12  # reservation ceiling only; real use is the sum of the case horizons


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
    package = ModuleType("scripts")
    package.__path__ = [str(ROOT / "scripts")]
    sys.modules["scripts"] = package


def install_feedforward(nominal, feedforward: float, np) -> None:
    """effective_yaw = ff * command + K * (command - measured), same clip and wheel kinematics.

    With feedforward == 1 this is the frozen law bit for bit; with zero yaw command the
    extra term vanishes, so non-turning tasks are untouched.
    """
    import math

    original = nominal.nominal_targets
    result_type = None

    def patched(command, state, *, ground_height_m=0.0):
        nonlocal result_type
        base = original(command, state, ground_height_m=ground_height_m)
        result_type = type(base)
        yaw = float(state.base_rpy[2])
        lateral_world = np.asarray((-math.sin(yaw), math.cos(yaw), 0.0))
        lateral = state.foot_offset_world @ lateral_world
        cmd = command.yaw_rate_rps
        effective = float(
            np.clip(
                feedforward * cmd
                + nominal.yaw_feedback_gain * (cmd - state.base_angular_velocity_body[2]),
                -nominal.yaw_request_limit_rps,
                nominal.yaw_request_limit_rps,
            )
        )
        wheels = np.clip((command.forward_velocity_mps - effective * lateral) / 0.087, -30.0, 30.0)
        return result_type(
            base.nominal_joint_target_rad, wheels, base.leg_extension_target_m, effective
        )

    nominal.nominal_targets = patched
    return patched


def keep_record_consistent(controller, feedforward: float) -> None:
    """The frozen record formula knows nothing about the feedforward; restate its two yaw fields."""
    compute = controller.compute

    def wrapped(command, state, ground, action):
        torque = compute(command, state, ground, action)
        cmd = float(command.yaw_rate_rps)
        value = feedforward * cmd + controller._nominal.yaw_feedback_gain * (
            cmd - float(state.base_angular_velocity_body[2])
        )
        controller.last_record["nominal_unclipped_yaw_rps"] = value
        controller.last_record["nominal_unlimited_yaw_rps"] = value
        controller.last_record["command_feedforward_gain"] = feedforward
        return torque

    controller.compute = wrapped


def worker(feedforward: float) -> int:
    folder = OUTPUT / f"FF{feedforward:g}"
    folder.mkdir(parents=True, exist_ok=True)
    h = load("yawgain44_helpers", ROOT / "scripts/b22_symmetry_physics43.py")
    configure_paths()
    backend = load("yawgain44_forbidden", ROOT / "scripts/b22_numpy_backend38.py")
    backend.deny_frameworks38()
    # isort: off
    import mujoco
    import mujoco._functions
    import mujoco._rollout
    import engine_binding  # noqa: F401
    from archive13.atomic_archive_13 import ArchiveWriter, AtomicCourseNativeGuard
    import numpy as np
    from controller18 import install_controller18
    from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
    from run_rl16_training_08 import _boundary
    from run_world_upright_reference_11 import _binding
    from world_upright_course_11 import WorldUprightCourseEnv

    from scripts import run_d1_latest_rl as common
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    # isort: on

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
    try:
        recipe, recorder, _ = h.recorder()
        api = h.api42()
        api.install_servo_counter42({"pure_servo_advance_max": CONTROLS_MAX})
        servo = api._servo_counter42
        library = read(MANIFEST)["runtime_environment"]["LD_PRELOAD"]
        with (
            RollingEngineRuntime(
                Path(library), control_limit=CONTROLS_MAX, construction_limit=2
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
            env.set_native_interval_reader(
                guard.interval_summary, begin_interval=guard.begin_interval
            )
            install_feedforward(controller._nominal, feedforward, np)
            keep_record_consistent(controller, feedforward)
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
            (folder / "heldout").mkdir()
            for case in CASES or recipe.ORDER:
                receipt, _ = recorder.record_case(
                    runtime,
                    guard,
                    env,
                    None,
                    recipe.make_schedule(case),
                    "zero",
                    folder / "heldout" / f"{case}_zero",
                    writer=writer,
                    pair_initial=None,
                    checkpoint_sha256=None,
                )
                result["cases"].append(
                    {
                        "case": case,
                        "record_valid": bool(receipt["record_valid"]),
                        "controls": receipt["completed_controls"],
                    }
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
        if writer.failed or writer.stop_requested or result.get("cleanup_errors"):
            result["status"] = "INCOMPLETE"
        (folder / "worker_report.json").write_text(json.dumps(result, indent=2))
    return int(result["status"] != "COMPLETE")


def derive_verifier(feedforward: float) -> None:
    """Register verify_control18 with exactly one changed line: the nominal unclipped yaw law.

    The frozen verifier recomputes `servo + 4*(servo - measured)` independently; the candidate
    law adds a command feedforward, so the verifier is derived with that single edit and
    everything else (clips, wheel kinematics, torque chain) is still checked unchanged.
    """
    path = ROOT / "runtime/d1_b22/research/continuation18/verify_control18.py"
    source = path.read_text()
    old = "unclipped_yaw = servo_yaw_rps + 4.0*(servo_yaw_rps-body_yaw_rate)"
    if source.count(old) != 1:
        raise RuntimeError("frozen verifier no longer contains the expected single yaw-law line")
    derived = source.replace(
        old, f"unclipped_yaw = {feedforward!r}*servo_yaw_rps + 4.0*(servo_yaw_rps-body_yaw_rate)"
    )
    module = ModuleType("verify_control18")
    module.__file__ = str(path)
    sys.modules["verify_control18"] = module
    exec(compile(derived, str(path), "exec"), module.__dict__)  # noqa: S102 -- single audited edit


def read_scores(feedforward: float) -> int:
    """Independent saved-data scoring through the 43 reader chain; no model, no physics."""
    folder = OUTPUT / f"FF{feedforward:g}"
    h = load("yawgain44_helpers", ROOT / "scripts/b22_symmetry_physics43.py")
    configure_paths()
    session = folder / "session.json"
    if not session.exists():
        # Host receipt for the saved-data reader: the same frozen source closure and contract id
        # as the completed 43 run (this exploratory worker executes only those frozen sources).
        reference = read(ROOT / "results/b22_symmetry43_physics_20261018_01/session.json")
        session.write_text(
            json.dumps(
                {
                    "execution_contract_id": reference["execution_contract_id"],
                    "source_hashes": reference["source_hashes"],
                    "provenance": "copied from the completed 43 session; exploratory run, not a qualification",
                }
            )
        )
    derive_verifier(feedforward)
    maths = h.saved_math()
    reader = maths.reader
    worker_result = read(folder / "worker_report.json")
    if worker_result["status"] != "COMPLETE" or worker_result["error"] is not None:
        raise RuntimeError("worker incomplete")
    construction = reader.document(folder / "construction_receipt.json")
    binding = reader.check_binding(construction["geometry_binding"])
    geometry = construction["compiled_geometry"]
    pure_helper = h.reader42()
    scores = []
    with pure_helper.geometry_counter42(sys.modules["geometry21"], CONTROLS_MAX):
        for case in CASES or maths.recipes.ORDER:
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
            _, canonical = reader.heldout_case(
                folder, spec, "zero", binding, geometry, expected_variant="combined", mirror=mirror
            )
            scores.append(maths.score.score_case(canonical, expected_seed=task.seed, mirror=mirror))
    (folder / "readback.json").write_text(
        json.dumps({"feedforward": feedforward, "cases": scores}, default=str)
    )
    return 0


def child_args() -> list[str]:
    return (
        ["--output", OUTPUT.name, "--cases", ",".join(CASES)]
        if CASES
        else ["--output", OUTPUT.name]
    )


def execute() -> int:
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    codes = {}
    for feedforward in FEEDFORWARDS:
        env = os.environ.copy()
        for key, value in read(MANIFEST)["runtime_environment"].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        with (OUTPUT / f"worker_FF{feedforward:g}.log").open("x") as log:
            child = subprocess.Popen(
                [
                    "/usr/bin/python3",
                    str(DRIVER),
                    "--worker",
                    "--feedforward",
                    str(feedforward),
                    *child_args(),
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
        codes[str(feedforward)] = child.returncode
        read_env = {k: v for k, v in env.items() if k != "LD_PRELOAD"}
        with (OUTPUT / f"read_FF{feedforward:g}.log").open("x") as log:
            reader_child = subprocess.run(
                [
                    "/usr/bin/python3",
                    str(DRIVER),
                    "--read",
                    "--feedforward",
                    str(feedforward),
                    *child_args(),
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
    global OUTPUT, CASES, FEEDFORWARDS
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    group.add_argument("--read", action="store_true")
    parser.add_argument("--feedforward", type=float, help="single setting for --worker/--read")
    parser.add_argument("--feedforwards", help="comma list for --execute")
    parser.add_argument(
        "--output", help="directory name under results/ (must be new for --execute)"
    )
    parser.add_argument(
        "--cases", help="comma list of preregistered case ids; default the six originals"
    )
    args = parser.parse_args()
    if args.output:
        OUTPUT = ROOT / "results" / args.output
    if args.cases:
        CASES = tuple(args.cases.split(","))
    if args.feedforwards:
        FEEDFORWARDS = tuple(float(x) for x in args.feedforwards.split(","))
    if args.read:
        return read_scores(args.feedforward)
    return worker(args.feedforward) if args.worker else execute()


if __name__ == "__main__":
    raise SystemExit(main())
