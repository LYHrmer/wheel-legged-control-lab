"""One prepaid world-upright short RL train and twelve matched heldout cases.

The root launcher alone reserves and starts this child. Import is stdlib-only;
MuJoCo, the isolated runtime, PPO and the actual model are loaded only after
the source/environment preflight inside ``main``. This worker never retries.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from scripts import run_d1_latest_rl as common

HERE = Path(__file__).resolve().parent
W = HERE.parent
SCHEMA = "d1-world-upright-short-rl16-training-heldout-11-v1"
CONTRACT_SHA = "e6e9d9d61d43d5eb3184c6c0cb31493f9f95e76cf4ab5632de02ce28135cd929"
TRAIN_CONTROLS = 65536
HELDOUT_RESERVED = 19600
TOTAL_RESERVED = 85136
NORMAL_NATIVE_RESERVED = 425680
CONTROL_RETURN_OFFSET_CLOSED_E = 777003


def _preflight(session: dict[str, Any], path: Path) -> Path:
    from budget_spec_11 import BudgetSpec

    budget = BudgetSpec(65536, 1024, 256, 4)
    expected = {
        "schema": SCHEMA,
        "phase": "T11",
        "run_id": "world_upright_short_rl11_01",
        "control_limit": TOTAL_RESERVED,
        "normal_native_limit": NORMAL_NATIVE_RESERVED,
        "compiler_native_limit": 2,
        "wallclock_limit_s": 1800,
        "training_control_limit": TRAIN_CONTROLS,
        "heldout_control_limit": HELDOUT_RESERVED,
        "segments": [TRAIN_CONTROLS, *([1600] * 10), 1800, 1800],
        "seed": 88621,
        "ppo_seed": 88621,
        "command_seed": 88622,
        "measurement_seed": 88623,
        "budget_spec": budget.as_dict(),
        "budget_spec_sha256": budget.canonical_sha256(),
        "short_plan_schema": "d1-world-upright-short1000-preregistered-plan-v1",
        "contract_sha256": CONTRACT_SHA,
        "retry_permitted": False,
    }
    if any(session.get(key) != value for key, value in expected.items()):
        raise RuntimeError("11-S session or full reserved budget differs from contract")
    if (session.get("argv") != sys.argv or sys.argv[1:] != ["--session", str(path)]
            or common._absolute(session, "session_path") != path
            or session.get("worker_argv", [])[-len(sys.argv):] != sys.argv):
        raise RuntimeError("11-S worker argv differs from unique launcher reservation")
    output = common._absolute(session, "output_directory")
    if not output.is_dir():
        raise RuntimeError("11-S output must be precreated exclusively by root")
    if type(session.get("wall_clock_utc")) is not str or not session["wall_clock_utc"]:
        raise RuntimeError("11-S launcher UTC clock is absent")
    sources = session.get("source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("11-S source closure is absent")
    for name, record in sources.items():
        source = Path(name)
        if (not source.is_absolute() or source.is_symlink() or not source.is_file()
                or not isinstance(record, dict)
                or source.stat().st_size != record.get("bytes")
                or common._sha256(source) != record.get("sha256")):
            raise RuntimeError("11-S frozen input differs: " + name)
    critical = (
        Path(__file__).resolve(), HERE / "rl16_learning_11.py",
        HERE / "budget_spec_11.py", HERE / "short_curriculum_11.py",
        HERE / "short_episode_plan_11.py", HERE / "short_plan_recipe_11.py",
        HERE / "short_heldout_recipe_11.py", HERE / "short_heldout_11.py",
        HERE / "short_corridor_11.py", HERE / "run_short_rl16_training_11.py",
        HERE / "next_short_training_contract_11.md",
        HERE / "short_seed_semantics_11.md", HERE / "astra_short_rl_go_11.json",
        HERE / "rl16_heldout_score_11.py",
        W / "upright11/world_upright_course_11.py",
        W / "upright11/run_world_upright_reference_11.py",
        W / "run_rl16_training_08.py",
        common._absolute(session, "library"),
        common._absolute(session, "continuation_import_repair"),
        common._absolute(session, "preregistered_scoring_path"),
        common._absolute(session, "binding_module"),
    )
    if any(str(source) not in sources for source in critical):
        raise RuntimeError("11-S critical source/library/scoring input absent from freeze")
    if session.get("short_plan_source_sha256") != sources[
        str(HERE / "short_plan_recipe_11.py")
    ]["sha256"]:
        raise RuntimeError("11-S plan source identity differs from session")
    for field in ("contract_documents", "dependency_hashes"):
        rows = session.get(field)
        if (not isinstance(rows, dict) or not rows
                or any(sources.get(name) != identity for name, identity in rows.items())):
            raise RuntimeError("11-S " + field + " is not in source closure")
    if session["preregistered_scoring_path"] not in session["contract_documents"]:
        raise RuntimeError("11-S scorer contract is not preregistered")
    if (common._sha256(common._absolute(session, "contract_path")) != CONTRACT_SHA
            or common._sha256(common._absolute(session, "library")) != common.DSO_SHA256):
        raise RuntimeError("11-S contract or isolated DSO identity differs")
    go_path = HERE / "astra_short_rl_go_11.json"
    go = json.loads(go_path.read_text())
    if (common._sha256(go_path) != session.get("go_sha256")
            or go.get("decision") != "GO" or go.get("contract_sha256") != CONTRACT_SHA
            or any(go.get(key) != expected[key] for key in (
                "control_limit", "normal_native_limit",
                "compiler_native_limit", "wallclock_limit_s",
            ))):
        raise RuntimeError("11-S exact reviewed source GO differs")
    for name, value in go["inputs"].items():
        if sources.get(name) != value:
            raise RuntimeError("11-S GO input absent from frozen source: " + name)
    for key, value in session["runtime_environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError("11-S isolated environment differs: " + key)
    if ("LD_LIBRARY_PATH" in os.environ or os.environ.get("LD_BIND_NOW") != "1"
            or os.environ.get("LD_PRELOAD") != session["library"]):
        raise RuntimeError("11-S isolated engine environment is invalid")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    session_path = args.session.resolve(strict=True)
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if not isinstance(session, dict):
        raise TypeError("11-S launcher session must be a JSON object")
    output = _preflight(session, session_path)
    common._save(output / "worker_preflight.json", {
        "schema": SCHEMA,
        "before_engine_or_learning_import": True,
        "frozen_input_count": len(session["source_hashes"]),
        "preregistered_scoring_path": session["preregistered_scoring_path"],
    })
    common._load_import_repair(session, output)
    if ("LD_LIBRARY_PATH" in os.environ or os.environ.get("LD_BIND_NOW") != "1"
            or os.environ.get("LD_PRELOAD") != session["library"]):
        raise RuntimeError("11-S import repair did not restore isolated engine environment")

    import engine_binding
    import mujoco
    from budget_spec_11 import BudgetSpec
    from course_native_guard_08 import CourseNativeGuard
    from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
    from rl16_learning_11 import WORLD_UPRIGHT_REFERENCE_DEFINITION
    from run_rl16_training_08 import (
        _audited_learning_origins,
        _boundary,
        _control_step_caller_verified,
    )
    from run_short_rl16_training_11 import train_short_phase
    from run_world_upright_reference_11 import _binding
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from short_heldout_11 import record_heldout_case
    from short_heldout_recipe_11 import ORDER, TASKS, heldout_schedule
    from short_plan_recipe_11 import PLAN_SCHEMA, SELECTION_SEED, build_plan
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    from world_upright_course_11 import WorldUprightCourseEnv

    if (PLAN_SCHEMA != session["short_plan_schema"]
            or SELECTION_SEED != session["command_seed"]
            or session.get("reference_definition") != WORLD_UPRIGHT_REFERENCE_DEFINITION
            or tuple(task.horizon for task in TASKS) != (1600,) * 5 + (1800,)):
        raise RuntimeError("11-S imported plan/reference/heldout identities differ")
    if Path(engine_binding.__file__).resolve() != common._absolute(session, "binding_module"):
        raise RuntimeError("11-S engine binding helper has another frozen origin")
    allowed = {W, W / "course_impl08", W / "upright11", HERE}
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name.endswith(("_08", "_11")) and getattr(module, "__file__", None):
            source = Path(module.__file__).resolve(strict=True)
            if source.parent not in allowed or str(source) not in session["source_hashes"]:
                raise RuntimeError("11-S imported course source differs: " + name)
            origins[name] = str(source)
    origins["engine_binding"] = str(Path(engine_binding.__file__).resolve())
    common._save(output / "module_origins.json", origins)
    cold = _nominal_geometry.cache_info()._asdict()
    if cold["misses"] != 0 or cold["currsize"] != 0:
        raise RuntimeError("11-S nominal geometry cache was not cold")
    common._save(output / "nominal_cache_cold.json", cold)

    warnings: list[str] = []

    def warning(message: str) -> None:
        warnings.append(str(message))
        raise RuntimeError("MuJoCo warning: " + str(message))

    def terminated(_sig: int, _frame: Any) -> None:
        raise common.TerminationRequested("11-S interrupted or 1800s wall limit reached")

    old_warning = mujoco.get_mju_user_warning()
    old_term = signal.signal(signal.SIGTERM, terminated)
    runtime = guard = env = None
    training_result = checkpoint_manifest = training_guard_segment = None
    cases: list[dict[str, Any]] = []
    failure = None
    model_calls = {"learn": 0, "save": 0, "load": 0, "predict": 0}
    model_phase = ["construction"]
    prediction_calls_by_phase = {}
    probe_batches = []

    def count_model_calls(frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        if code.co_name == "learn" and code.co_filename.endswith(
            "stable_baselines3/ppo/ppo.py"
        ):
            model_calls["learn"] += 1
        elif (code.co_name in ("save", "load", "predict")
              and code.co_filename.endswith("stable_baselines3/common/base_class.py")):
            model_calls[code.co_name] += 1
            if code.co_name == "predict":
                phase = model_phase[0]
                prediction_calls_by_phase[phase] = prediction_calls_by_phase.get(phase, 0) + 1
                if phase == "training_and_finalization":
                    observation = frame.f_locals["observation"]
                    probe_batches.append({
                        "shape": list(observation.shape),
                        "deterministic": frame.f_locals.get("deterministic"),
                    })
    if sys.getprofile() is not None:
        raise RuntimeError("11-S worker inherited an unexpected profile hook")
    sys.setprofile(count_model_calls)
    started = time.monotonic()
    try:
        mujoco.set_mju_user_warning(warning)
        with RollingEngineRuntime(
            Path(session["library"]), control_limit=TOTAL_RESERVED,
            construction_limit=2,
        ) as runtime:
            common._save(output / "runtime_initial.json", {
                "proof": runtime.proof, **_boundary(runtime),
            })
            with CourseNativeGuard(runtime) as guard:
                caps = QualifiedCommandCaps(1.6, 0.0, 0.0, 0.3, False)
                env = runtime.construct(lambda: WorldUprightCourseEnv(
                    caps=caps, command_source=lambda _tick, _time: FullDriveCommand(),
                    spawn_position_m=(-8.0, -4.7, 0.455),
                    max_steps=1000, mode="train",
                ))
                constructed = runtime.state()
                cache = _nominal_geometry.cache_info()._asdict()
                if (constructed["construction_attempts"] != 2
                        or constructed["construction_returns"] != 2
                        or cache["misses"] != 1 or cache["currsize"] != 1
                        or env.pure_test_components):
                    raise RuntimeError("11-S construction was not course+nominal cold2")
                common._save(output / "construction_receipt.json", {
                    "C_state": constructed,
                    "nominal_cache": cache,
                    "course_plant_model_address": int(env.plant.model._address),
                    "course_plant_data_address": int(env.plant.data._address),
                    "compiled_geometry": env.plant.collision_terrain_metadata,
                    "actual_geometry_binding": _binding(env.plant),
                    "action_schema": env.controller.action_schema,
                    "control_schema": env.controller.control_schema,
                    "world_upright_reference_definition": WORLD_UPRIGHT_REFERENCE_DEFINITION,
                })
                env.set_native_interval_reader(
                    guard.interval_summary, begin_interval=guard.begin_interval,
                )
                budget = BudgetSpec.from_dict(session["budget_spec"])
                plan = build_plan(source_sha256=session["short_plan_source_sha256"])
                model_phase[0] = "training_and_finalization"
                (policy, checkpoint_manifest, training_result,
                 training_guard_segment, curriculum) = train_short_phase(
                    session, runtime, guard, env, output,
                    budget=budget, plan=plan, caps=caps,
                    ppo_seed=session["ppo_seed"],
                    measurement_seed=session["measurement_seed"],
                )
                env.mode = "eval"
                env.controller.mode = "eval"
                common._save(output / "learning_dependency_origins_before_heldout.json",
                             _audited_learning_origins(session))
                common._save(output / "heldout_mode_transition.json", {
                    "same_compiled_model_data_pair": [
                        int(env.plant.model._address), int(env.plant.data._address),
                    ],
                    "from": "train", "to": "eval",
                    "numerical_controller_unchanged": True,
                    "training_partial_episode_preserved": True,
                    "boundary": _boundary(runtime),
                })
                for name in ORDER:
                    schedule = heldout_schedule(name)
                    pair_initial = None
                    for actor in ("zero", "final_policy"):
                        folder = output / "heldout" / f"{name}_{actor}"
                        folder.parent.mkdir(exist_ok=True)
                        model_phase[0] = "heldout:" + name + ":" + actor
                        receipt, initial = record_heldout_case(
                            runtime, guard, env, policy, schedule, actor, folder,
                            pair_initial=pair_initial,
                        )
                        cases.append(receipt)
                        common._save(output / f"heldout_progress_{len(cases):02d}.json", {
                            "case": name, "actor": actor,
                            "completed_cases": len(cases),
                            "actual_training_controls": curriculum.completed_controls,
                            "case_receipt": str(folder / "case_receipt.json"),
                            "record_valid": receipt["record_valid"],
                            "boundary": _boundary(runtime),
                        })
                        if not receipt["record_valid"]:
                            raise RuntimeError("11-S heldout record invalid; no later case may run")
                        if actor == "zero":
                            pair_initial = initial
                    if pair_initial is None:
                        raise RuntimeError("11-S heldout pair lacks first initial state")
                model_phase[0] = "closure"
                expected_predicts = sum(row["policy_predictions"] for row in cases)
                if (model_calls != {
                        "learn": 1, "save": 1, "load": 2, "predict": 4 + expected_predicts,
                    } or prediction_calls_by_phase.get("training_and_finalization") != 4
                        or len(probe_batches) != 4
                        or any(row["shape"] != [32, 99] or row["deterministic"] is not True
                               for row in probe_batches)
                        or any(prediction_calls_by_phase.get(
                            "heldout:" + row["case_id"] + ":" + row["actor"], 0,
                        ) != row["policy_predictions"] for row in cases)):
                    raise RuntimeError("11-S actual learn/save/load/probe/heldout call counts differ")
                state = runtime.state()
                actual_controls = TRAIN_CONTROLS + sum(row["completed_controls"] for row in cases)
                if (len(cases) != 12 or warnings
                        or runtime.control_completed != actual_controls
                        or runtime.control_attempted != runtime.control_completed
                        or runtime.returned != 5 * runtime.control_completed
                        or runtime.attempted != runtime.returned
                        or runtime.failed != 0
                        or runtime.advanced_substeps != runtime.returned
                        or guard.returned != runtime.returned
                        or guard.attempted != guard.returned
                        or guard.checked != guard.returned
                        or guard.failure is not None
                        or state["control_returns"] != 5 * actual_controls
                        or state["control_attempts"] != state["control_returns"]
                        or state["construction_attempts"] != 2
                        or state["construction_returns"] != 2
                        or state["native_construction_caller_verified"] is not True
                        or not _control_step_caller_verified(
                            state, runtime.proof, session["source_hashes"], engine_binding,
                        )
                        or state["ccd_attempts"] != state["ccd_returns"]
                        or (state["ccd_attempts"] > 0
                            and state["native_ccd_caller_verified"] is not True)
                        or state["violations"] != 0
                        or runtime.ledger_receipt()["forbidden_entries"] != 0
                        or _nominal_geometry.cache_info().misses != 1):
                    raise RuntimeError("11-S final C/Python/guard counts differ")
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        sys.setprofile(None)
        signal.signal(signal.SIGTERM, old_term)
        mujoco.set_mju_user_warning(old_warning)
        if runtime is not None:
            runtime.unbind()
        final_state = None if runtime is None else runtime.state()
        caller_verified = (
            False if final_state is None else _control_step_caller_verified(
                final_state, runtime.proof, session["source_hashes"], engine_binding,
            )
        )
        complete = (
            failure is None and training_result is not None
            and training_result["qualified_for_final_checkpoint"] is True
            and checkpoint_manifest is not None and len(cases) == 12
            and all(row["record_valid"] for row in cases)
            and not warnings and caller_verified
            and final_state is not None and final_state["phase"] == 0
            and final_state["target_model"] == 0
            and final_state["target_data"] == 0
        )
        common._save(output / "worker_receipt.json", {
            "schema": SCHEMA, "failure": failure, "warnings": warnings,
            "execution_complete": complete,
            "training_receipt": None if training_result is None
            else str(output / "training" / "learning_receipt.json"),
            "training_guard_segment": training_guard_segment,
            "actual_model_calls": model_calls,
            "actual_prediction_calls_by_phase": prediction_calls_by_phase,
            "actual_final_probe_batches": probe_batches,
            "training_policy_forward_is_separate_from_BaseAlgorithm_predict": True,
            "final_checkpoint_manifest": None if checkpoint_manifest is None
            else str(output / "final_checkpoint_manifest.json"),
            "heldout_cases": [
                {key: row[key] for key in (
                    "case_id", "actor", "seed", "terrain", "control_cap",
                    "completed_controls", "policy_prediction_attempts",
                    "policy_predictions", "record_valid", "terminated",
                    "truncated", "stop_reason",
                )} for row in cases
            ],
            "control_step_caller_verified_against_closed_E_callsite": caller_verified,
            "C_final": final_state,
            "python": None if runtime is None else runtime.ledger_receipt(),
            "native_guard": None if guard is None else guard.report(),
            "elapsed_wall_s": time.monotonic() - started,
            "retry_permitted": False,
            "full_physical_qualification_pending_independent_scoring": True,
            "RL_contribution_pending_matched_pair_scoring": True,
        })
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
