"""One bounded C34 fixed velocity-reference and distance campaign."""
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
from selection34 import best34, select34, selected_stage_b_cases34

CONTRACT = "C34_fixed_velocity_reference_and_40mm_v1"


def origins34(sources):
    result = runtime_origins30(sources)
    for name in ("runtime31", "ledger31", "record31", "macro31", "task31",
                 "observation31", "side_skill31", "body_curve33", "side_timing33",
                 "pair_prefix33", "task34", "macro34", "record34", "selection34",
                 "side_velocity34"):
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError("C34 actual module is missing: "+name)
        path = str(Path(module.__file__).resolve(strict=True))
        actual = identity(path)
        if sources.get(path) != actual:
            raise RuntimeError("C34 actual module escaped frozen source: "+name)
        result[name] = dict(path=path, **actual)
    path = str(Path(__file__).resolve(strict=True))
    if sources.get(path) != identity(path):
        raise RuntimeError("C34 worker escaped frozen source")
    result["worker34"] = dict(path=path, **identity(path))
    return result


def bootstrap34(session, output):
    symbols = bootstrap30(session, output)
    import side_skill31  # noqa: F401 - actual source origin frozen before construction
    import observation31  # noqa: F401
    import macro31  # noqa: F401
    import task31  # noqa: F401
    import task34  # noqa: F401
    import macro34  # noqa: F401
    import record34  # noqa: F401
    import side_velocity34  # noqa: F401
    from runtime31 import SideAccess31, NativeWindow31, EpisodeBudget31, install_reset_yaw31
    from ledger31 import ModelLedger31
    from record34 import recorder34
    from side_timing33 import make_side33
    from pair_prefix33 import pair_prefix33
    from side_velocity34 import make_side34
    additions = locals().copy()
    additions.pop("symbols")
    symbols.update(additions)
    symbols["origins34"] = origins34(session["source_hashes"])
    write(output/"runtime34_module_origins_before.json", symbols["origins34"])
    return symbols


def preflight34(session, session_path, output):
    expected = (22000, 10, 40, 900, 960, 1020, 1200)
    seen = tuple(session[k] for k in ("control_limit", "cycles_limit", "macros_limit",
                                      "soft_s", "close_s", "hard_s", "outer_s"))
    limits = session["model_limits"]
    wanted = dict(load=1, torch_load=3, predict=6001, forward=0,
                  evaluate_actions=0, predict_values=0, backward=0,
                  learn=0, train=0, save=0)
    if (session["execution_contract_id"] != CONTRACT or seen != expected
            or session["numerical_protocol_contract_id"] != CONTRACT
            or session["outer_execution_contract_id"] != CONTRACT
            or session["arm"] != "development" or session["mode"] != "fixed"
            or session["render"] is not False or session["retry_permitted"] is not False
            or session["argv"] != sys.argv or limits != wanted
            or Path(session["output_directory"]).resolve() != output
            or session["spawn_position_m"] != [-8., -4.7, .455]
            or session["seed"] != 271001 or session["side_arm"] != "teacher"):
        raise RuntimeError("C34 finite fixed session differs")
    if (session["stage_A_cases"] != [
            dict(beta=b, case_id=f"d030_beta{tag}_{side}", direction=d,
                 distance_m=.03, initial_yaw_rad=0.)
            for b, tag in ((.5,"050"),(1.,"100"))
            for side,d in (("left",1),("right",-1))]
            or session["stage_B_zero_cases"] != [
            dict(beta=0., case_id=f"d040_beta000_{side}", direction=d,
                 distance_m=.04, initial_yaw_rad=0.)
            for side,d in (("left",1),("right",-1))]
            or session["alpha"] != 1.
            or session["allowed_beta34"] != [0., .5, 1.]
            or session["allowed_distance34_m"] != [.03, .04]):
        raise RuntimeError("C34 frozen development case order differs")
    if session["static_API_global_caps"] != dict(copy=15160, forward=1689440,
            fullM=15000, jac=60000, jacBody=1447680, objectVelocity=30010):
        raise RuntimeError("C34 static API ceilings differ")
    pythonpath = session["runtime_environment"].get("PYTHONPATH", "")
    if (pythonpath.split(os.pathsep)[0] !=
            "/home/lyh/.local/lib/python3.10/site-packages"):
        raise RuntimeError("C34 bootstrap requires installed MuJoCo site-packages first")
    for path, expected_identity in session["source_hashes"].items():
        if identity(path) != expected_identity:
            raise RuntimeError("C34 source changed: "+path)
    for key, expected_value in session["runtime_environment"].items():
        if os.environ.get(key) != expected_value:
            raise RuntimeError("C34 environment differs: "+key)
    contract = Path(session["contract34_path"]).resolve(strict=True)
    if session["source_hashes"].get(str(contract)) != identity(contract):
        raise RuntimeError("C34 frozen contract missing from source closure")
    baselines34(session)
    archived33(session)
    write(output/"preflight_ready.json", dict(pid=os.getpid(),
          session_sha256=identity(session_path)["sha256"], monotonic_s=time.monotonic()))


def baselines34(session):
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


def score34(task, macro, baseline, *, cancel, same_task_zero=None):
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
    if same_task_zero is not None:
        gates.update(
            goal_error_vs_d40_beta0=task["final_xy_error_m"] <=
                same_task_zero["final_xy_error_m"]+.002,
            retention_vs_d40_beta0=task["retention_ratio"] is not None and
                task["retention_ratio"] >= max(.90,
                    same_task_zero["retention_ratio"]-.02),
            tau2_vs_d40_beta0=cost <= 1.10*same_task_zero["full_tau2_mean"])
    common["gates"] = gates
    common["eligible"] = all(gates.values())
    common["nominal_commanded_speed_mps"] = None
    return common


def archived33(session):
    run = Path(session["archived_c33_run_path"]).resolve(strict=True)
    report_path = Path(session["archived_c33_report_path"]).resolve(strict=True)
    selection_path = run/"campaign_selection33.json"
    sources = session["source_hashes"]
    for path in (report_path, selection_path):
        if sources.get(str(path)) != identity(path):
            raise RuntimeError("C34 archived C33 evidence escaped source closure")
    report = json.loads(report_path.read_text())
    selection = json.loads(selection_path.read_text())
    if (report.get("record_valid") is not True or report.get("run") != str(run)
            or report.get("selected_cancel_pair_qualified") is not True
            or selection.get("selected_alpha33") != 1.):
        raise RuntimeError("C34 archived D30 beta0 is not accepted C33 alpha1")
    by_id = {row["case_id"]: row for row in selection["all_attempted_cases"]}
    accepted = report["attempted_cases"]
    pair = []
    for side in ("left", "right"):
        case_id = session["archived_c33_cases"][side]
        cancel_id = session["archived_c33_cancel_cases"][side]
        row = by_id[case_id]
        if (case_id != f"alpha_100_{side}" or cancel_id != f"cancel_100_{side}"
                or row["metrics"]["eligible"] is not True
                or accepted[case_id]["record_valid"] is not True
                or accepted[cancel_id]["record_valid"] is not True
                or accepted[case_id]["independent_learning_eligible"] is not True):
            raise RuntimeError("C34 archived D30 beta0 pair differs")
        receipt = run/f"episode_{row['cycle_index']}"/"cycle_receipt.json"
        if (sources.get(str(receipt)) != identity(receipt)
                or row["cycle_receipt_identity"] != identity(receipt)):
            raise RuntimeError("C34 archived D30 receipt escaped source closure")
        pair.append(row)
    return {"report": str(report_path), "report_identity": identity(report_path),
            "selection": str(selection_path), "selection_identity": identity(selection_path),
            "rows": pair, "cancel_case_ids": session["archived_c33_cancel_cases"]}


def pair34(left, right, *, distance_m, beta, archived=False):
    if (left["direction"], right["direction"]) != (1, -1):
        raise RuntimeError("C34 pair directions differ")
    metrics = (left["metrics"], right["metrics"])
    return {"distance_m": distance_m, "beta": beta,
        "case_ids": [left["case_id"], right["case_id"]],
        "archived": archived, "complete": True,
        "eligible": all(row["eligible"] for row in metrics),
        "max_side_duration_s": max(row["side_duration_s"] for row in metrics),
        "mean_full_tau2_mean": sum(row["full_tau2_mean"] for row in metrics)/2}


def run_cases34(s, session, output, runtime, guard, env, hybrid, policy,
                writer, window, access, clock, yaw_state, stop):
    budget = s["EpisodeBudget31"](cycles=10, controls=22000, macros=40)
    clock["budget34"] = budget
    baseline = baselines34(session)
    archived = archived33(session)
    writer.commit_json(output/"baseline34.json", baseline)
    writer.commit_json(output/"archived_c33_reference34.json", archived)
    results, definitions = [], []
    pairs = [pair34(*archived["rows"], distance_m=.03, beta=0., archived=True)]
    predictions = 0
    stop_reason = "development_complete"
    d40_zero = {}

    def run_one(template, *, kind="development"):
        nonlocal predictions
        index = len(results)
        if stop.is_set() or writer.stop_requested:
            raise RuntimeError("C34 stopped before next reserved case")
        reservation = budget.begin(controls=runtime.control_completed, native=runtime.returned)
        case = dict(session, **template, kind=kind, cancel=kind=="cancel",
                    cycle_index=index,
                    control_limit=reservation["control_limit"])
        clock["beta34"] = template["beta"]
        clock["distance34"] = template["distance_m"]
        recorder = s["recorder34"](template["distance_m"])
        result, cycle = recorder(case, runtime, guard, env, hybrid, policy,
            writer, window, clock, stop, output, cycle_index=index,
            global_control_offset=reservation["global_control_begin"],
            access=access, yaw_state=yaw_state)
        closed = bool(cycle and cycle.get("terminal_kind") in
                      ("success", "controlled_failure", "safe_cancel"))
        actual = budget.finish(controls=runtime.control_completed, native=runtime.returned,
                               macros=result["actual_event_transitions"], complete=closed)
        writer.commit_json(output/f"budget_cycle_{index:02d}.json", actual)
        if not closed or result.get("fatal_failure"):
            raise RuntimeError("C34 incomplete/unsafe cycle; no retry or rescue")
        predictions += result["policy_predictions"]
        if predictions > 6000:
            raise RuntimeError("C34 B22 control prediction cap exceeded")
        folder = output/f"episode_{index}"
        receipt_path = folder/"cycle_receipt.json"
        task = result["online_task"]
        macro = json.loads((folder/"macro_transitions31.json").read_text())
        side = "left" if template["direction"] == 1 else "right"
        bridge = s["pair_prefix33"](session["baseline_paths33"][side]["fixed"],
                                    folder, session["source_hashes"])
        writer.commit_json(output/f"entry_bridge34_{index:02d}.json", bridge)
        if not all(task["entry_gates"].values()):
            raise RuntimeError("C34 original entry stability gate failed")
        same = d40_zero.get(side) if template["distance_m"]==.04 and template["beta"] else None
        score = score34(task, macro, baseline[side], cancel=kind=="cancel",
                        same_task_zero=same)
        finished_goal = kind != "cancel" and task["online_success"] is True
        retained = (task["signed_lateral_m"]*task["retention_ratio"]
                    if task["retention_ratio"] is not None else None)
        score["signed_lateral_m"] = task["signed_lateral_m"]
        score["retained_signed_lateral_m"] = retained
        score["nominal_commanded_speed_mps"] = (template["distance_m"] /
            score["side_duration_s"] if finished_goal and score["side_duration_s"] else None)
        score["actual_retained_progress_per_cycle_mps"] = (retained /
            score["side_duration_s"] if finished_goal and retained is not None
            and score["side_duration_s"] else None)
        row = {"cycle_index": index, "case_id": template["case_id"],
               "distance_m": template["distance_m"], "beta": template["beta"],
               "body_beta34": template["beta"], "body_alpha33": 1.,
               "incoming_hybrid_distance_m": .03,
               "requested_distance_m": template["distance_m"],
               "applied_fast_distance_m": template["distance_m"],
               "direction": template["direction"],
               "kind": kind, "terminal_kind": cycle["terminal_kind"],
               "cycle_receipt_identity": identity(receipt_path),
               "metrics": score, "actual_controls": actual["actual_controls"],
               "policy_predictions": result["policy_predictions"]}
        writer.commit_json(output/f"case34_{index:02d}.json", row)
        definitions.append({key: row[key] for key in (
            "cycle_index", "case_id", "distance_m", "beta", "direction", "kind",
            "incoming_hybrid_distance_m", "requested_distance_m",
            "applied_fast_distance_m", "cycle_receipt_identity")})
        results.append(row)
        clock["last_case34"] = row
        writer.commit_json(output/f"progress34_{index:02d}.json", {
            "closed_cases": len(results), "controls": runtime.control_completed,
            "normal_native": runtime.returned, "macros": budget.macros,
            "policy_predictions": predictions, "last_case": row})
        return row

    def run_stage(templates):
        nonlocal stop_reason
        for template in templates:
            row = run_one(template)
            if row["terminal_kind"] == "controlled_failure" or not row["metrics"]["eligible"]:
                stop_reason = "controlled_task_or_quality_failure"
                return False
            if template["distance_m"] == .04 and template["beta"] == 0.:
                d40_zero["left" if template["direction"] == 1 else "right"] = row["metrics"]
            if template["direction"] == -1:
                left, right = results[-2:]
                if (left["distance_m"], left["beta"], left["direction"],
                    right["distance_m"], right["beta"], right["direction"]) != (
                    template["distance_m"], template["beta"], 1,
                    template["distance_m"], template["beta"], -1):
                    raise RuntimeError("C34 nonadjacent setting pair")
                pairs.append(pair34(left, right, distance_m=template["distance_m"],
                                    beta=template["beta"]))
        return True

    stage_a_complete = run_stage(session["stage_A_cases"])
    selected_a = best34([p for p in pairs if p["distance_m"] == .03])
    if stage_a_complete:
        stage_b_complete = run_stage(session["stage_B_zero_cases"])
        if stage_b_complete:
            templates = selected_stage_b_cases34(selected_a["beta"])
            if templates:
                run_stage(templates)
    selected = select34(pairs)
    by_id = {row["case_id"]: row for row in archived["rows"]+results}
    c33_times = {side: archived["rows"][index]["metrics"]["side_duration_s"]
                 for index, side in enumerate(("left", "right"))}
    d40_zero_pair = next((p for p in pairs if p["distance_m"] == .04
                          and p["beta"] == 0.), None)
    pair_reports = []
    for pair in pairs:
        left, right = (by_id[name] for name in pair["case_ids"])
        cycle_times = {"left": left["metrics"]["side_duration_s"],
                       "right": right["metrics"]["side_duration_s"]}
        milestone = None
        if pair["distance_m"] == .03:
            milestone = all(cycle_times[side] <= .95*c33_times[side]
                            for side in ("left", "right"))
        elif pair["distance_m"] == .04:
            milestone = all(.04/cycle_times[side] >= 1.25*(.03/c33_times[side])
                            for side in ("left", "right"))
        same_task_gain = None
        if pair["distance_m"] == .04 and pair["beta"] and d40_zero_pair:
            zero_times = {side: by_id[name]["metrics"]["side_duration_s"]
                          for side, name in zip(("left", "right"),
                                                d40_zero_pair["case_ids"])}
            same_task_gain = all(cycle_times[side] <= .95*zero_times[side]
                                 for side in ("left", "right"))
        pair_reports.append({**pair, "side_cycle_s": cycle_times,
            "nominal_D_over_T_mps": {side: pair["distance_m"]/cycle_times[side]
                                       for side in ("left", "right")},
            "distance_specific_milestone": milestone,
            "d40_vs_beta0_five_percent_faster": same_task_gain})
    if selected is not None and not selected["archived"]:
        tag = {0.: "000", .5: "050", 1.: "100"}[selected["beta"]]
        dist = "030" if selected["distance_m"] == .03 else "040"
        for side, direction in (("left", 1), ("right", -1)):
            row = run_one(dict(case_id=f"cancel_d{dist}_beta{tag}_{side}",
                beta=selected["beta"], distance_m=selected["distance_m"],
                direction=direction, initial_yaw_rad=0.), kind="cancel")
            if row["terminal_kind"] != "safe_cancel" or not row["metrics"]["eligible"]:
                raise RuntimeError("C34 selected-setting cancellation did not safely close")
    selection = {"schema": "d1-c34-campaign-selection-v1", "candidates": pair_reports,
                 "all_attempted_cases": results, "selected_stage_A_beta": selected_a["beta"],
                 "selected_setting34": selected, "stop_reason": stop_reason,
                 "archived_C33_reference": archived,
                 "cancel_conditions": {"directions": [1, -1],
                     "reused_archived_cancel": bool(selected["archived"]),
                     "next_control_after_first_dual_velocity_overlap": True},
                 "qualification": False, "independent_readback_pending": True}
    writer.commit_json(output/"campaign_selection34.json", selection)
    writer.commit_json(output/"case_definition34.json", {
        "schema": "d1-c34-case-definition-v1", "cases": definitions,
        "selected_setting34": selected, "stop_reason": stop_reason})
    return {"closed_cases": len(results), "controls": runtime.control_completed,
            "normal_native": runtime.returned, "true_macros": budget.macros,
            "policy_predictions": predictions, "selected_setting34": selected,
            "stop_reason": stop_reason, "budget": {"limits": budget.limits,
            "closed": budget.closed, "macros": budget.macros}}


def execute34(session, session_path, output):
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
        preflight34(session, session_path, output)
        signal.setitimer(signal.ITIMER_REAL, session["soft_s"])
        s = symbols = bootstrap34(session, output)
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
            with runtime_type(Path(session["library"]), control_limit=22000,
                              construction_limit=2) as runtime:
                with s["AtomicCourseNativeGuard"](runtime, writer) as guard:
                    env = runtime.construct(lambda: s["WorldUprightCourseEnv"](
                        caps=s["QualifiedCommandCaps"](1.6, 0., 0., .3, False),
                        command_source=lambda *_: s["FullDriveCommand"](),
                        spawn_position_m=session["spawn_position_m"],
                        max_steps=2200, mode="eval"))
                    binding = s["capture_binding24"](env.plant)
                    clock = dict(control_index=0, cycle_index=0, global_offset=0,
                                 beta34=0., distance34=.03)
                    hybrid = s["install_hybrid27"](env, side_arm="teacher",
                        side_factory=lambda plant: s["make_side34"](plant,
                            beta=0., configured_distance_m=.03, geometry_binding=binding,
                            control_index_provider=lambda: clock["control_index"],
                            cycle_index_provider=lambda: clock["cycle_index"]))
                    original_reset = env.reset

                    def reset_with_case34(*args, **kwargs):
                        value = original_reset(*args, **kwargs)
                        hybrid.side.set_case34(beta=clock["beta34"],
                            configured_distance_m=clock["distance34"])
                        return value

                    env.reset = reset_with_case34
                    access = s["SideAccess31"](mj, runtime, env.plant,
                        hybrid.side.scratch, cycles_limit=10)
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
                        raise RuntimeError("C34 one cold construction count differs")
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
                        raise RuntimeError("C34 strict B22 probe changed physics/policy")
                    writer.commit_json(output/"strict_load.json", load_report)
                    counter.phase = "control"
                    result = run_cases34(s, session, output, runtime, guard, env,
                        hybrid, B22, writer, window, access, clock, yaw_state, stop)
                    caller = s["_control_step_caller_verified"](runtime.state(),
                        runtime.proof, session["source_hashes"], s["engine_binding"])
                    access_report = access.report()
                    static_caps = session["static_API_global_caps"]
                    if (not caller or runtime.control_attempted != runtime.control_completed
                            or runtime.returned != 5*runtime.control_completed
                            or runtime.returned > 110000
                            or runtime.attempted != runtime.returned or runtime.thread_violations
                            or guard.checked != runtime.returned
                            or not access_report["within_actual_bounds"]
                            or any(access_report["counts"][key] > cap
                                   for key, cap in static_caps.items())
                            or access.rejected or copy_edges["rejected"]):
                        raise RuntimeError("C34 cumulative physical/static/copy ledger differs")
                    counts = counter.summary()["counts"]
                    expected = dict(load=1, torch_load=3,
                        predict=result["policy_predictions"]+1,
                        forward=0, evaluate_actions=0, predict_values=0,
                        backward=0, learn=0, train=0, save=0)
                    if any(counts.get(key, {}).get("returned", 0) != value
                           for key, value in expected.items()):
                        raise RuntimeError("C34 B22 model ledger differs")
                    if origins34(session["source_hashes"]) != s["origins34"]:
                        raise RuntimeError("C34 module origins changed during campaign")
                    writer.commit_json(output/"runtime34_module_origins.json", s["origins34"])
                    writer.commit_json(output/"runtime31_module_origins.json", s["origins34"])
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
        receipt = dict(schema="d1-c34-headless-worker-v1",
            execution_contract_id=CONTRACT, arm=session["arm"],
            session_identity=identity(session_path), failure=failure,
            cleanup_errors=cleanup, warnings=warnings, result=result,
            C_final=None if runtime is None else runtime.state(),
            python=None if runtime is None else runtime.ledger_receipt(),
            native_guard=None if guard is None else guard.report(),
            model_calls=None if counter is None else counter.summary(),
            partial_budget=None if "budget34" not in clock else dict(
                limits=clock["budget34"].limits, closed=clock["budget34"].closed,
                open=clock["budget34"].open, macros=clock["budget34"].macros),
            last_case=clock.get("last_case34"),
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
    return execute34(json.loads(path.read_text()), path, args.output.resolve(strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
