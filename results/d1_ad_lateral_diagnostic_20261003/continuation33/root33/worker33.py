"""One bounded C33 fixed-timing campaign on the inherited C31 physical owner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

from worker30 import bootstrap30, identity, runtime_origins30, write

CONTRACT = "C33_fixed_S_curve_feasibility_v1"
ALPHAS = (0.85, 0.925, 1.0)


def origins33(sources):
    result = runtime_origins30(sources)
    for name in ("runtime31", "ledger31", "record31", "macro31", "task31",
                 "observation31", "side_skill31", "body_curve33", "side_timing33",
                 "pair_prefix33"):
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError("C33 actual module is missing: "+name)
        path = str(Path(module.__file__).resolve(strict=True))
        actual = identity(path)
        if sources.get(path) != actual:
            raise RuntimeError("C33 actual module escaped frozen source: "+name)
        result[name] = dict(path=path, **actual)
    path = str(Path(__file__).resolve(strict=True))
    if sources.get(path) != identity(path):
        raise RuntimeError("C33 worker escaped frozen source")
    result["worker33"] = dict(path=path, **identity(path))
    return result


def bootstrap33(session, output):
    symbols = bootstrap30(session, output)
    import side_skill31  # noqa: F401 - actual source origin frozen before construction
    import observation31  # noqa: F401
    import macro31  # noqa: F401
    import task31  # noqa: F401
    from runtime31 import SideAccess31, NativeWindow31, EpisodeBudget31, install_reset_yaw31
    from ledger31 import ModelLedger31
    from record31 import record_case31
    from side_timing33 import make_side33
    from pair_prefix33 import pair_prefix33
    additions = locals().copy()
    additions.pop("symbols")
    symbols.update(additions)
    symbols["origins33"] = origins33(session["source_hashes"])
    write(output/"runtime33_module_origins_before.json", symbols["origins33"])
    return symbols


def preflight33(session, session_path, output):
    expected = (17600, 8, 32, 900, 960, 1020, 1200)
    seen = tuple(session[k] for k in ("control_limit", "cycles_limit", "macros_limit",
                                      "soft_s", "close_s", "hard_s", "outer_s"))
    limits = session["model_limits"]
    wanted = dict(load=1, torch_load=3, predict=4801, forward=0,
                  evaluate_actions=0, predict_values=0, backward=0,
                  learn=0, train=0, save=0)
    if (session["execution_contract_id"] != CONTRACT or seen != expected
            or session["arm"] != "development" or session["mode"] != "fixed"
            or session["render"] is not False or session["retry_permitted"] is not False
            or session["argv"] != sys.argv or limits != wanted
            or Path(session["output_directory"]).resolve() != output
            or session["spawn_position_m"] != [-8., -4.7, .455]
            or session["seed"] != 271001 or session["side_arm"] != "teacher"):
        raise RuntimeError("C33 finite fixed session differs")
    if session["development_cases"] != [
            {"case_id": case["case_id"], "alpha": case["alpha"],
             "direction": case["direction"], "initial_yaw_rad": case["initial_yaw_rad"]}
            for case in cases33()]:
        raise RuntimeError("C33 frozen development case order differs")
    if session["static_API_global_caps"] != dict(copy=12128, forward=1351552,
            fullM=12000, jac=48000, jacBody=1158144, objectVelocity=24008):
        raise RuntimeError("C33 static API ceilings differ")
    for path, expected_identity in session["source_hashes"].items():
        if identity(path) != expected_identity:
            raise RuntimeError("C33 source changed: "+path)
    for key, expected_value in session["runtime_environment"].items():
        if os.environ.get(key) != expected_value:
            raise RuntimeError("C33 environment differs: "+key)
    write(output/"preflight_ready.json", dict(pid=os.getpid(),
          session_sha256=identity(session_path)["sha256"], monotonic_s=time.monotonic()))


def baselines33(session):
    result = {}
    paths = session["baseline_paths33"]
    for side in ("left", "right"):
        result[side] = {}
        for mode in ("fixed", "zero"):
            path = Path(paths[side][mode]).resolve(strict=True)
            if session["source_hashes"].get(str(path)) != identity(path):
                raise RuntimeError("C33 baseline escaped source hash: "+str(path))
            document = json.loads(path.read_text())
            if document.get("record_valid") is not True:
                raise RuntimeError("C33 original baseline is invalid")
            task = document["task"]
            components = document["macros"]["full_scene_components"]
            result[side][mode] = {
                "side_duration_s": float(task["side_duration_s"]),
                "final_xy_error_m": float(task["final_xy_error_m"]),
                "retention_ratio": float(task["retention_ratio"]),
                "full_tau2_mean": float(components["torque_normalized_square_mean"]),
                "source_path": str(path), "source_identity": identity(path)}
    return result


def score33(task, macro, baseline, *, cancel):
    cost_integral = float(macro["full_case_components"]["torque_normalized_square_integral_s"])
    elapsed = float(macro["full_case_components"]["elapsed_s"])
    cost = cost_integral/elapsed
    common = {"full_tau2_mean": cost, "full_tau2_integral_s": cost_integral,
              "full_elapsed_s": elapsed, "side_duration_s": task["side_duration_s"],
              "final_xy_error_m": task["final_xy_error_m"],
              "retention_ratio": task["retention_ratio"],
              "safety_complete": task["safety_complete"],
              "entry_gates": task["entry_gates"]}
    if cancel:
        common["eligible"] = bool(task["online_safe_cancel"] and task["safety_complete"])
        common["cancel_to_handoff_controls"] = task["cancel_to_handoff_controls"]
        return common
    fixed, zero = baseline["fixed"], baseline["zero"]
    gates = {
        "original_safety_and_task": bool(task["online_success"] and task["safety_complete"]),
        "entry": all(task["entry_gates"].values()),
        "goal_error_vs_zero": task["final_xy_error_m"] <= zero["final_xy_error_m"]+.002,
        "retention_vs_zero": task["retention_ratio"] is not None and
            task["retention_ratio"] >= max(.90, zero["retention_ratio"]-.02),
        "tau2_vs_fixed": cost <= 1.10*fixed["full_tau2_mean"],
        "tau2_vs_zero": cost <= 1.20*zero["full_tau2_mean"]}
    common["gates"] = gates
    common["eligible"] = all(gates.values())
    common["nominal_fixed_engineering_gain"] = (
        task["side_duration_s"] <= .95*fixed["side_duration_s"] and
        task["side_duration_s"] <= .90*zero["side_duration_s"])
    return common


def cases33():
    for alpha, label in zip(ALPHAS, ("085", "0925", "100")):
        for side, direction in (("left", 1), ("right", -1)):
            yield dict(case_id=f"alpha_{label}_{side}", alpha=alpha,
                       direction=direction, kind="development", cancel=False,
                       initial_yaw_rad=0.)


def run_cases33(s, session, output, runtime, guard, env, hybrid, policy,
                writer, window, access, clock, yaw_state, stop):
    budget = s["EpisodeBudget31"](cycles=8, controls=17600, macros=32)
    clock["budget33"] = budget
    baseline = baselines33(session)
    writer.commit_json(output/"baseline33.json", baseline)
    results, definitions, pairs = [], [], []
    predictions = 0
    stop_reason = "development_complete"

    def run_one(template):
        nonlocal predictions
        index = len(results)
        if stop.is_set() or writer.stop_requested:
            raise RuntimeError("C33 stopped before next reserved case")
        reservation = budget.begin(controls=runtime.control_completed, native=runtime.returned)
        case = dict(session, **template, cycle_index=index,
                    control_limit=reservation["control_limit"])
        clock["alpha33"] = template["alpha"]
        result, cycle = s["record_case31"](case, runtime, guard, env, hybrid, policy,
            writer, window, clock, stop, output, cycle_index=index,
            global_control_offset=reservation["global_control_begin"],
            access=access, yaw_state=yaw_state)
        closed = bool(cycle and cycle.get("terminal_kind") in
                      ("success", "controlled_failure", "safe_cancel"))
        actual = budget.finish(controls=runtime.control_completed, native=runtime.returned,
                               macros=result["actual_event_transitions"], complete=closed)
        writer.commit_json(output/f"budget_cycle_{index:02d}.json", actual)
        if not closed or result.get("fatal_failure"):
            raise RuntimeError("C33 incomplete/unsafe cycle; no retry or rescue")
        predictions += result["policy_predictions"]
        if predictions > 4800:
            raise RuntimeError("C33 B22 control prediction cap exceeded")
        folder = output/f"episode_{index}"
        receipt_path = folder/"cycle_receipt.json"
        task = result["online_task"]
        macro = json.loads((folder/"macro_transitions31.json").read_text())
        side = "left" if template["direction"] == 1 else "right"
        bridge = s["pair_prefix33"](session["baseline_paths33"][side]["fixed"],
                                    folder, session["source_hashes"])
        writer.commit_json(output/f"entry_bridge33_{index:02d}.json", bridge)
        if not all(task["entry_gates"].values()):
            raise RuntimeError("C33 original entry stability gate failed")
        score = score33(task, macro, baseline[side], cancel=template["cancel"])
        row = {"cycle_index": index, "case_id": template["case_id"],
               "alpha": template["alpha"], "body_alpha33": template["alpha"],
               "direction": template["direction"],
               "kind": template["kind"], "terminal_kind": cycle["terminal_kind"],
               "cycle_receipt_identity": identity(receipt_path),
               "metrics": score, "actual_controls": actual["actual_controls"],
               "policy_predictions": result["policy_predictions"]}
        writer.commit_json(output/f"case33_{index:02d}.json", row)
        definitions.append({key: row[key] for key in ("cycle_index", "case_id", "alpha",
                         "direction", "kind", "cycle_receipt_identity")})
        results.append(row)
        clock["last_case33"] = row
        writer.commit_json(output/f"progress33_{index:02d}.json", {
            "closed_cases": len(results), "controls": runtime.control_completed,
            "normal_native": runtime.returned, "macros": budget.macros,
            "policy_predictions": predictions, "last_case": row})
        return row

    for template in cases33():
        row = run_one(template)
        if row["terminal_kind"] == "controlled_failure":
            stop_reason = "controlled_task_failure_closes_higher_alpha"
            break
        if not row["metrics"]["eligible"]:
            stop_reason = "development_gate_failure_closes_higher_alpha"
            break
        if template["direction"] == -1:
            left, right = results[-2:]
            if (left["alpha"], left["direction"], right["alpha"], right["direction"]) != (
                    template["alpha"], 1, template["alpha"], -1):
                raise RuntimeError("C33 direction pair is not adjacent and common-alpha")
            pairs.append({"alpha": template["alpha"], "case_ids": [left["case_id"],
                right["case_id"]], "eligible": True,
                "max_side_duration_s": max(left["metrics"]["side_duration_s"],
                                           right["metrics"]["side_duration_s"]),
                "mean_full_tau2_mean": (left["metrics"]["full_tau2_mean"]+
                                         right["metrics"]["full_tau2_mean"])/2})
    winner = min(pairs, key=lambda pair: (pair["max_side_duration_s"],
                 pair["mean_full_tau2_mean"], pair["alpha"])) if pairs else None
    selected = None if winner is None else winner["alpha"]
    if winner is not None:
        label = {0.85: "085", 0.925: "0925", 1.0: "100"}[selected]
        for side, direction in (("left", 1), ("right", -1)):
            row = run_one(dict(case_id=f"cancel_{label}_{side}", alpha=selected,
                               direction=direction, kind="cancel", cancel=True,
                               initial_yaw_rad=0.))
            if row["terminal_kind"] != "safe_cancel" or not row["metrics"]["eligible"]:
                raise RuntimeError("C33 selected-alpha cancellation did not safely close")
    selection = {"schema": "d1-c33-campaign-selection-v1", "candidates": pairs,
                 "all_attempted_cases": results, "selected_alpha33": selected,
                 "stop_reason": stop_reason, "cancel_conditions": {
                     "directions": [1, -1] if selected is not None else [],
                     "selected_alpha": selected,
                     "next_control_after_first_dual_velocity_overlap": True},
                 "qualification": False, "independent_readback_pending": True}
    writer.commit_json(output/"campaign_selection33.json", selection)
    writer.commit_json(output/"case_definition33.json", {
        "schema": "d1-c33-case-definition-v1", "cases": definitions,
        "selected_alpha33": selected, "stop_reason": stop_reason})
    return {"closed_cases": len(results), "controls": runtime.control_completed,
            "normal_native": runtime.returned, "true_macros": budget.macros,
            "policy_predictions": predictions, "selected_alpha33": selected,
            "stop_reason": stop_reason, "budget": {"limits": budget.limits,
            "closed": budget.closed, "macros": budget.macros}}


def execute33(session, session_path, output):
    stop = threading.Event()
    failure = None
    runtime = guard = env = counter = writer = access = symbols = None
    original_copy = previous_warning = copy_edges = None
    warnings, cleanup, result, clock = [], [], {}, {}
    started = time.monotonic()

    def stop_request(*_):
        stop.set()
        if writer is not None:
            writer.request_stop()

    previous_term = signal.signal(signal.SIGTERM, stop_request)
    previous_alarm = signal.signal(signal.SIGALRM, stop_request)
    try:
        preflight33(session, session_path, output)
        signal.setitimer(signal.ITIMER_REAL, session["soft_s"])
        s = symbols = bootstrap33(session, output)
        common, mj = s["common"], s["mujoco"]

        class JsonWriter(s["ArchiveWriter"]):
            def commit_json(self, path, value):
                return super().commit_json(path, common._jsonable(value))

        writer = JsonWriter()

        def warning(message):
            warnings.append(str(message))
            stop.set()
            raise RuntimeError("MuJoCo warning: "+str(message))

        previous_warning = mj.get_mju_user_warning()
        mj.set_mju_user_warning(warning)
        runtime_type = s["side_owner_runtime_type"](stop)
        counter = s["ModelLedger31"](False, session["model_limits"])
        with counter:
            writer.commit_json(output/"mapped_libraries_before_load.json",
                               s["mapped_libraries"](session["source_hashes"]))
            with runtime_type(Path(session["library"]), control_limit=17600,
                              construction_limit=2) as runtime:
                with s["AtomicCourseNativeGuard"](runtime, writer) as guard:
                    env = runtime.construct(lambda: s["WorldUprightCourseEnv"](
                        caps=s["QualifiedCommandCaps"](1.6, 0., 0., .3, False),
                        command_source=lambda *_: s["FullDriveCommand"](),
                        spawn_position_m=session["spawn_position_m"],
                        max_steps=2200, mode="eval"))
                    binding = s["capture_binding24"](env.plant)
                    clock = dict(control_index=0, cycle_index=0, global_offset=0,
                                 alpha33=ALPHAS[0])
                    hybrid = s["install_hybrid27"](env, side_arm="teacher",
                        side_factory=lambda plant: s["make_side33"](plant,
                            alpha=ALPHAS[0], geometry_binding=binding,
                            control_index_provider=lambda: clock["control_index"],
                            cycle_index_provider=lambda: clock["cycle_index"]))
                    original_reset = env.reset

                    def reset_with_alpha(*args, **kwargs):
                        value = original_reset(*args, **kwargs)
                        hybrid.side.set_alpha33(clock["alpha33"])
                        return value

                    env.reset = reset_with_alpha
                    access = s["SideAccess31"](mj, runtime, env.plant,
                        hybrid.side.scratch, cycles_limit=8)
                    access.side = hybrid.side
                    hybrid.attach_side_access(access)
                    runtime.register_side_access(access)
                    access.install()
                    original_copy, copy_edges = s["fence_copy_data30"](mj,
                        plant=env.plant, side_access=access, stop_event=stop)
                    window = s["NativeWindow31"](binding)
                    guard.native_sink = window.sink
                    yaw_state = s["install_reset_yaw31"](env)
                    if (runtime.state()["construction_returns"] != 2 or
                            s["_nominal_geometry"].cache_info().misses != 1):
                        raise RuntimeError("C33 one cold construction count differs")
                    writer.commit_json(output/"construction_receipt.json", dict(
                        C_state=runtime.state(), nominal_cache=s["_nominal_geometry"]
                            .cache_info()._asdict(),
                        compiled_geometry=env.plant.collision_terrain_metadata,
                        geometry_binding=s["_binding"](env.plant),
                        side_kinematic_binding=binding,
                        skill30_joint_limits=dict(jnt_range=env.plant.model.jnt_range.tolist(),
                          jnt_limited=env.plant.model.jnt_limited.tolist()),
                        side_profile=asdict(hybrid.side.cfg), proof=runtime.proof,
                        boundary=s["_boundary"](runtime), data_addresses=dict(
                          live=int(env.plant.data._address),
                          measurement=int(env.plant.measurement_data._address),
                          side_scratch=int(hybrid.side.scratch._address))))
                    env.set_native_interval_reader(guard.interval_summary,
                                                   begin_interval=guard.begin_interval)
                    counter.phase = "probe"
                    before_load = s["_boundary"](runtime)
                    B22, load_report = s["load_and_verify_final"](
                        session["checkpoint_folder"], session["checkpoint_manifest"],
                        return_model=True)
                    if (s["_boundary"](runtime) != before_load or
                            load_report["probe_actions_byte_exact"] is not True or
                            load_report["probe_rows"] != 32):
                        raise RuntimeError("C33 strict B22 probe changed physics/policy")
                    writer.commit_json(output/"strict_load.json", load_report)
                    counter.phase = "control"
                    result = run_cases33(s, session, output, runtime, guard, env,
                        hybrid, B22, writer, window, access, clock, yaw_state, stop)
                    caller = s["_control_step_caller_verified"](runtime.state(),
                        runtime.proof, session["source_hashes"], s["engine_binding"])
                    access_report = access.report()
                    static_caps = session["static_API_global_caps"]
                    if (not caller or runtime.control_attempted != runtime.control_completed
                            or runtime.returned != 5*runtime.control_completed
                            or runtime.returned > 88000
                            or runtime.attempted != runtime.returned or runtime.thread_violations
                            or guard.checked != runtime.returned
                            or not access_report["within_actual_bounds"]
                            or any(access_report["counts"][key] > cap
                                   for key, cap in static_caps.items())
                            or access.rejected or copy_edges["rejected"]):
                        raise RuntimeError("C33 cumulative physical/static/copy ledger differs")
                    counts = counter.summary()["counts"]
                    expected = dict(load=1, torch_load=3,
                        predict=result["policy_predictions"]+1,
                        forward=0, evaluate_actions=0, predict_values=0,
                        backward=0, learn=0, train=0, save=0)
                    if any(counts.get(key, {}).get("returned", 0) != value
                           for key, value in expected.items()):
                        raise RuntimeError("C33 B22 model ledger differs")
                    if origins33(session["source_hashes"]) != s["origins33"]:
                        raise RuntimeError("C33 module origins changed during campaign")
                    writer.commit_json(output/"runtime33_module_origins.json", s["origins33"])
                    writer.commit_json(output/"runtime31_module_origins.json", s["origins33"])
                    writer.commit_json(output/"reset_yaw_receipt31.json", yaw_state)
                    writer.commit_json(output/"loaded_origins_final.json",
                                       s["_audited_learning_origins"](session))
                    writer.commit_json(output/"mapped_libraries_final.json",
                                       s["mapped_libraries"](session["source_hashes"]))
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        print(failure["traceback"], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for name, action in (
            ("restore_copy", lambda: setattr(symbols["mujoco"], "mj_copyData", original_copy)
             if symbols is not None and original_copy is not None else None),
            ("restore_static_queries", lambda: access.restore() if access is not None else None),
            ("restore_warning", lambda: symbols["mujoco"].set_mju_user_warning(previous_warning)
             if symbols is not None else None)):
            try:
                action()
            except BaseException as error:
                cleanup.append(name+": "+repr(error))
        receipt = dict(schema="d1-c33-headless-worker-v1",
            execution_contract_id=CONTRACT, arm=session["arm"],
            session_identity=identity(session_path), failure=failure,
            cleanup_errors=cleanup, warnings=warnings, result=result,
            C_final=None if runtime is None else runtime.state(),
            python=None if runtime is None else runtime.ledger_receipt(),
            native_guard=None if guard is None else guard.report(),
            model_calls=None if counter is None else counter.summary(),
            partial_budget=None if "budget33" not in clock else dict(
                limits=clock["budget33"].limits, closed=clock["budget33"].closed,
                open=clock["budget33"].open, macros=clock["budget33"].macros),
            last_case=clock.get("last_case33"),
            side_access=None if access is None else access.report(),
            copy_edges=copy_edges, archive_failed=None if writer is None else writer.failed,
            elapsed_wall_s=time.monotonic()-started, retry_permitted=False,
            execution_complete=failure is None and not cleanup, qualification=False)
        write(output/"worker_receipt.json", receipt if symbols is None else
              symbols["common"]._jsonable(receipt))
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGALRM, previous_alarm)
    return 0 if receipt["execution_complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.session.resolve(strict=True)
    return execute33(json.loads(path.read_text()), path, args.output.resolve(strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
