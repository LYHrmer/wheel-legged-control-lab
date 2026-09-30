"""Summarize C18 saved readbacks and closed receipts without running physics.

The C18 primary six pairs alone determine the RL contribution result. Mirror
remains a separate robustness check. Outputs are exclusive and never infer a
pass from a planned budget or a changed seed label.
"""

from __future__ import annotations

import builtins
import csv
import hashlib
import json
from pathlib import Path
import sys
import time


_IMPORT = builtins.__import__
FORBIDDEN = frozenset(("torch", "mujoco", "stable_baselines3", "gym", "gymnasium",
                       "wheel_legged_control", "engine_binding"))


def guarded_import(name, *args, **kwargs):
    if name.partition(".")[0] in FORBIDDEN:
        raise RuntimeError("model/physics import denied: " + name)
    return _IMPORT(name, *args, **kwargs)


builtins.__import__ = guarded_import
import numpy as np


C = Path(__file__).resolve().parent.parent
OLD = C.parent / "continuation17"
SUMMARY = C / "qualification_summary_18.json"
TABLE = C / "task_results_18.csv"
CASES = ("flat_0p6", "flat_1p6", "flat_1p2_yaw", "bumps_0p4",
         "rough_0p35", "ramp_0p45_complete")
ARRAYS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
SOURCE_NAMES = ("controller18.py", "residual18.py", "verify_control18.py",
                "read_eval18.py", "score18.py", "recipes18.py", "host18.py",
                "worker18.py", "eval18.py", "floor_bridge18.py",
                "offline_floor18.py", "qualification_contract_18.md",
                "qualification_spec_18.json", "pure_tests_receipt_18.json",
                "development_readback_18.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path: Path, consumed: dict) -> dict:
    require(path.is_file() and not path.is_symlink(), "missing/linked input: " + str(path))
    path = path.resolve(strict=True)
    item = {"sha256": digest(path), "bytes": path.stat().st_size}
    consumed[str(path)] = item
    return item


def document(path: Path, consumed: dict) -> dict:
    identity(path, consumed)
    item = json.loads(path.read_text())
    require(isinstance(item, dict), "non-object JSON: " + str(path))
    return item


def in_plan(path: Path, plan: dict, consumed: dict) -> dict:
    item = identity(path, consumed)
    require(plan["inputs"][str(path.resolve())] == item,
            "frozen input identity differs: " + str(path))
    return item


def pair_initials(a: Path, b: Path, consumed: dict) -> dict:
    a_id, b_id = identity(a, consumed), identity(b, consumed)
    with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
        require(set(ARRAYS) <= set(left) and set(ARRAYS) <= set(right),
                "initial state is missing required arrays")
        fields = {name: bool(left[name].shape == right[name].shape
                             and left[name].dtype == right[name].dtype
                             and left[name].tobytes(order="C") == right[name].tobytes(order="C"))
                  for name in ARRAYS}
    require(all(fields.values()), "zero/grouped initial state differs: " + str(a))
    return {"all_five_bitwise_equal": True, "fields": fields,
            "zero_identity": a_id, "grouped_identity": b_id}


def gate(score: dict) -> dict:
    required = ("record", "safety", "speed", "yaw", "final_window", "stopping",
                "terrain_channel", "ramp_geometry", "rl_terms", "task_reasons")
    require(all(key in score for key in required)
            and type(score.get("task_passed")) is bool
            and isinstance(score["task_reasons"], list), "task gate is incomplete")
    result = {"task_passed": score["task_passed"], "task_reasons": score["task_reasons"],
              "completed_controls": score["completed_controls"],
              "record_valid": score["record"]["record_valid"],
              "full_horizon": score["record"]["full_horizon"],
              "safety_passed": score["safety"]["passed"],
              "speed_passed": score["speed"]["passed"],
              "yaw_passed": score["yaw"]["passed"],
              "final_window_passed": score["final_window"]["passed"],
              "stopping_passed": score["stopping"]["passed"],
              "terrain_channel_passed": score["terrain_channel"]["passed"],
              "ramp_geometry_passed": score["ramp_geometry"]["passed"],
              "hold_mean_com_vx_mps": score["speed"].get("mean_com_vx_mps"),
              "hold_vx_rms_mps": score["speed"].get("rms_com_vx_error_vs_target_mps"),
              "hold_yaw_rms_rps": score["yaw"].get("rms_yaw_rate_error_vs_applied_servo_rps"),
              "drive_sse": score["rl_terms"]["sse_total"],
              "normalized_torque_squared_cost_sum": score["rl_terms"]["torque_cost_sum"]}
    if score["case_id"] == "ramp_0p45_complete":
        ramp = score["ramp_geometry"]
        result["ramp_positive_wheel_load_by_geom"] = ramp[
            "positive_wheel_load_native_counts_by_geom"]
        result["wheel_collision_world_x_min_m_by_index"] = ramp[
            "wheel_collision_world_x_min_m_by_index"]
        result["all_ramp_world_x_max_m"] = ramp["all_ramp_world_x_max_m"]
    return result


def read_arm(arm: str, plan: dict, plan_path: Path, consumed: dict) -> tuple[dict, dict]:
    spec = plan["arms"][arm]
    run = C / spec["output_directory"]
    readback_path = C / ("qualification_" + arm + "_readback_18.json")
    readback = document(readback_path, consumed)
    reader_host = document(readback_path.with_name(readback_path.stem + "_host_receipt.json"), consumed)
    session = document(run / "session.json", consumed)
    host = document(run / "host_receipt.json", consumed)
    worker = document(run / "worker_receipt.json", consumed)
    plan_identity = identity(plan_path, consumed)
    readback_identity = consumed[str(readback_path.resolve())]
    require(readback["schema"] == "d1-world-upright-stage18-independent-eval-readback-v1"
            and readback["run"] == str(run.resolve())
            and readback["controller_variant"] == "combined"
            and readback["source_closure_verified"] is True
            and readback["checkpoint_ZIP_verified"] is True
            and readback["physical_calls_performed_by_reader"] == 0
            and readback["model_loaded_by_reader"] is False
            and reader_host["failure"] is None and reader_host["exit_code"] == 0
            and reader_host["elapsed_s"] <= 300
            and reader_host["controls"] == reader_host["model_calls"] == 0
            and reader_host["output_identity"] == readback_identity
            and reader_host["reader_identity"] == plan["inputs"][str(C / "read_eval18.py")]
            and host["failure"] is None and host["exit_code"] == 0
            and host["cleanup_errors"] == [] and host["changed_sources"] == []
            and host["postcheck_complete"] is True
            and host["no_live_owned_processes"] is True
            and host["fully_reserved_budget_closed"] is True
            and worker["schema"] == "d1-control-repair-worker-18-v1"
            and worker["arm"] == arm and worker["failure"] is None
            and worker["execution_complete"] is True
            and worker["cleanup_errors"] == [] and worker["warnings"] == []
            and session["plan_identity"] == plan_identity
            and session["source_hashes"][str(plan_path)] == plan_identity
            and session["case_specs"] == spec["case_specs"]
            and session["floor_seed"] == spec["floor_seed"]
            and session["controller_variant"] == "combined",
            "worker/readback/host/session closure differs: " + arm)
    actual = readback["actual_completed_controls"]
    require(0 <= actual <= spec["controls"]
            and readback["actual_normal_native_returns"] == 5*actual
            and readback["actual_compiler_native_returns"] == 2
            and worker["result"]["completed_controls"] == actual
            and worker["C_final"]["construction_attempts"] == 2
            and worker["C_final"]["construction_returns"] == 2
            and worker["C_final"]["control_attempts"] == 5*actual
            and worker["C_final"]["control_returns"] == 5*actual
            and worker["python"]["control_completed"] == actual
            and worker["python"]["native_returned"] == 5*actual
            and worker["native_guard"]["native_checked"] == 5*actual,
            "actual control/native/compiler budget differs: " + arm)
    counts = worker["model_calls"]["counts"]
    predict = 1 + readback["floor"]["completed_controls"] + sum(
        row["predict_calls"] for row in readback["heldout_summaries"])
    expected = {"load": 1, "torch_load": 3, "predict": predict,
                "save": 0, "learn": 0, "train": 0, "forward": 0,
                "evaluate_actions": 0, "predict_values": 0, "backward": 0}
    require(all(counts.get(key, {}).get("attempted", 0)
                == counts.get(key, {}).get("returned", 0) == value
                for key, value in expected.items()), "model API calls differ: " + arm)
    rows = predict + 31  # 32-row probe replaces the first single predict row.
    require(counts["predict"]["rows_attempted"] == rows
            and counts["predict"]["rows_returned"] == rows
            and counts["predict"]["phases"]["probe"]["rows"] == 32
            and worker["model_calls"]["actor_rows"] == rows
            and worker["model_calls"]["critic_rows"] == 0,
            "model row ledger differs: " + arm)
    require(readback["floor"]["numeric_gate_passed"] is True,
            "floor numeric gate failed: " + arm)
    ledger = {"reserved_controls": spec["controls"], "actual_controls": actual,
              "reserved_normal_native": 5*spec["controls"],
              "actual_normal_native": 5*actual, "compiler_native_returns": 2,
              "policy_load_calls": 1, "torch_load_calls": 3,
              "predict_api_calls": predict, "actor_rows": rows, "critic_rows": 0,
              "training_calls": 0, "model_api_counts": expected,
              "worker_closed": True, "host_closed": True, "independent_reader_closed": True}
    return readback, ledger


def score_table(readback: dict, spec: dict, run: Path, consumed: dict,
                paired: bool) -> tuple[dict, dict]:
    expected = {(item["case_id"], actor) for item in spec["case_specs"]
                for actor in item["actors"]}
    scores = readback["numeric_scores"]
    found = {(row["case_id"], row["experiment_actor"]) for row in scores}
    require(len(scores) == len(expected) and found == expected,
            "missing/duplicate scored task")
    table, initial = {}, {}
    for item in spec["case_specs"]:
        case_id = item["case_id"]
        actors = item["actors"]
        require(actors == (["zero", "grouped_continue"] if paired else ["grouped_continue"]),
                "actor pairing differs: " + case_id)
        table[case_id] = {}
        for actor in actors:
            row = next(row for row in scores if row["case_id"] == case_id
                       and row["experiment_actor"] == actor)
            require(row["seed"] == item["seed"] and row["mirror"] is item["mirror"],
                    "scored task seed/mirror differs")
            table[case_id][actor] = gate(row)
        if paired:
            folder = run / "heldout"
            initial[case_id] = pair_initials(
                folder / f"{case_id}_zero" / "initial_state.npz",
                folder / f"{case_id}_grouped_continue" / "initial_state.npz", consumed)
    return table, initial


def contribution(pair: dict, scores: dict, spec: dict) -> dict:
    require(pair["schema"] == "d1-course-world-upright-rl16-heldout-pure-pair-18-v1"
            and pair["pair_count"] == 6 and pair["case_count"] == 12
            and len(pair["pairs"]) == 6, "six-task RL pair block differs")
    require([r["case_id"] for r in pair["pairs"]] == [
        r["case_id"] for r in spec["case_specs"]], "primary pair order differs")
    for row, item in zip(pair["pairs"], spec["case_specs"]):
        zero = scores[item["case_id"]]["zero"]
        policy = scores[item["case_id"]]["grouped_continue"]
        require(row["seed"] == item["seed"]
                and row["zero_task_passed"] is zero["task_passed"]
                and row["policy_task_passed"] is policy["task_passed"]
                and row["sse_zero"] == zero["drive_sse"]
                and row["sse_policy"] == policy["drive_sse"],
                "RL pair detached from scored tasks")
    return {"RL_contribution_passed": pair["RL_contribution_passed"],
            "all_six_policy_task_qualified": pair["all_six_policy_task_qualified"],
            "prerequisites": pair["prerequisites"], "cost_gate": pair["cost_gate"],
            "branch_a_task_conversion": pair["branch_a_task_conversion"],
            "branch_b_sse_improvement": pair["branch_b_sse_improvement"],
            "per_task": pair["pairs"],
            "torque_cost_is_normalized_squared_proxy_not_energy": True}


def development(plan_c: dict, consumed: dict) -> tuple[dict, dict]:
    plan_b_path = C / "plan_go_B18.json"
    plan_b = document(plan_b_path, consumed)
    require(plan_b["schema"] == "d1-control-repair-plan-18-v1"
            and plan_b["status"] == "GO"
            and set(plan_b["arms"]) == {"development"}, "development plan differs")
    require(plan_c["inputs"][str(plan_b_path)] == consumed[str(plan_b_path)],
            "qualification plan did not freeze development plan")
    spec = plan_b["arms"]["development"]
    run = C / spec["output_directory"]
    readback_path = C / "development_readback_18.json"
    readback = document(readback_path, consumed)
    reader_host = document(C / "development_readback_18_host_receipt.json", consumed)
    host = document(run / "host_receipt.json", consumed)
    worker = document(run / "worker_receipt.json", consumed)
    session = document(run / "session.json", consumed)
    require(plan_c["inputs"][str(readback_path)] == consumed[str(readback_path)]
            and readback["schema"] == "d1-world-upright-stage18-independent-eval-readback-v1"
            and readback["run"] == str(run.resolve())
            and readback["controller_variant"] == "combined"
            and readback["source_closure_verified"] is True
            and readback["checkpoint_ZIP_verified"] is True
            and readback["physical_calls_performed_by_reader"] == 0
            and readback["model_loaded_by_reader"] is False
            and readback["floor"]["numeric_gate_passed"] is True
            and readback["six_task_zero_pair"] is None
            and reader_host["failure"] is None and reader_host["exit_code"] == 0
            and reader_host["elapsed_s"] <= 300
            and reader_host["controls"] == reader_host["model_calls"] == 0
            and reader_host["output_identity"] == consumed[str(readback_path)]
            and reader_host["reader_identity"] == plan_b["inputs"][str(C / "read_eval18.py")]
            and host["failure"] is None and host["exit_code"] == 0
            and host["cleanup_errors"] == [] and host["changed_sources"] == []
            and host["postcheck_complete"] is True
            and host["no_live_owned_processes"] is True
            and host["fully_reserved_budget_closed"] is True
            and worker["schema"] == "d1-control-repair-worker-18-v1"
            and worker["arm"] == "development"
            and worker["failure"] is None and worker["execution_complete"] is True
            and worker["cleanup_errors"] == [] and worker["warnings"] == []
            and session["plan_identity"] == consumed[str(plan_b_path)]
            and session["source_hashes"][str(plan_b_path)] == consumed[str(plan_b_path)]
            and session["case_specs"] == spec["case_specs"]
            and session["floor_seed"] == spec["floor_seed"]
            and session["controller_variant"] == "combined",
            "development readback/receipt closure differs")
    actual = readback["actual_completed_controls"]
    require(0 <= actual <= spec["controls"]
            and readback["actual_normal_native_returns"] == 5*actual
            and readback["actual_compiler_native_returns"] == 2
            and worker["result"]["completed_controls"] == actual
            and worker["C_final"]["construction_attempts"] == 2
            and worker["C_final"]["construction_returns"] == 2
            and worker["C_final"]["control_attempts"] == 5*actual
            and worker["C_final"]["control_returns"] == 5*actual
            and worker["python"]["control_completed"] == actual
            and worker["python"]["native_returned"] == 5*actual
            and worker["native_guard"]["native_checked"] == 5*actual,
            "development actual counters differ")
    calls = worker["model_calls"]["counts"]
    predict = 1 + readback["floor"]["completed_controls"] + sum(
        row["predict_calls"] for row in readback["heldout_summaries"])
    expected_calls = {"load": 1, "torch_load": 3, "predict": predict,
                      "save": 0, "learn": 0, "train": 0, "forward": 0,
                      "evaluate_actions": 0, "predict_values": 0, "backward": 0}
    require(all(calls.get(key, {}).get("attempted", 0)
                == calls.get(key, {}).get("returned", 0) == value
                for key, value in expected_calls.items())
            and calls["predict"]["rows_attempted"] == predict+31
            and calls["predict"]["rows_returned"] == predict+31
            and calls["predict"]["phases"]["probe"]["rows"] == 32
            and worker["model_calls"]["actor_rows"] == predict+31
            and worker["model_calls"]["critic_rows"] == 0,
            "development model calls/rows differ")
    scores, _ = score_table(readback, spec, run, consumed, paired=False)
    require(set(scores) == {"rough_0p35", "ramp_0p45_complete"},
            "development scenario set differs")
    ledger = {"reserved_controls": spec["controls"], "actual_controls": actual,
              "reserved_normal_native": 5*spec["controls"],
              "actual_normal_native": 5*actual, "compiler_native_returns": 2,
              "policy_load_calls": 1, "torch_load_calls": 3,
              "predict_api_calls": predict, "actor_rows": predict+31,
              "critic_rows": 0, "training_calls": 0}
    return scores, ledger


def sum_ledgers(items: dict) -> dict:
    fields = ("reserved_controls", "actual_controls", "reserved_normal_native",
              "actual_normal_native", "compiler_native_returns", "policy_load_calls",
              "torch_load_calls", "predict_api_calls", "actor_rows", "critic_rows",
              "training_calls")
    return {field: sum(item[field] for item in items.values()) for field in fields}


def csv_rows(development_scores: dict, scores: dict, old_scores: dict) -> list[dict]:
    rows = []
    for arm, table in (("development", development_scores),
                       ("primary", scores["primary"]), ("mirror", scores["mirror"])):
        for case_id, actors in table.items():
            for actor, item in actors.items():
                old = old_scores.get(arm, {}).get(case_id, {}).get(actor, {})
                row = {"arm": arm, "case_id": case_id, "actor": actor,
                       "task_passed": item["task_passed"],
                       "task_reasons_json": json.dumps(item["task_reasons"]),
                       "completed_controls": item["completed_controls"]}
                for field in ("record_valid", "full_horizon", "safety_passed",
                              "speed_passed", "yaw_passed", "final_window_passed",
                              "stopping_passed", "terrain_channel_passed",
                              "ramp_geometry_passed", "hold_mean_com_vx_mps",
                              "hold_vx_rms_mps", "hold_yaw_rms_rps", "drive_sse",
                              "normalized_torque_squared_cost_sum"):
                    row[field] = item[field]
                row["C17_task_passed"] = old.get("task_passed")
                row["C17_hold_vx_rms_mps"] = old.get("hold_vx_rms_mps")
                row["C18_minus_C17_hold_vx_rms_mps"] = (
                    item["hold_vx_rms_mps"]-old["hold_vx_rms_mps"]
                    if item["hold_vx_rms_mps"] is not None
                    and old.get("hold_vx_rms_mps") is not None else None)
                rows.append(row)
    return rows


def main() -> None:
    started = time.monotonic()
    require(not SUMMARY.exists() and not TABLE.exists(), "exclusive output already exists")
    consumed = {}
    plan_path = C / "plan_go_C18.json"
    plan = document(plan_path, consumed)
    require(plan["schema"] == "d1-control-repair-plan-18-v1"
            and plan["status"] == "GO"
            and plan["development_precondition_verified"] is True
            and set(plan["arms"]) == {"primary", "mirror"},
            "C18 qualification plan is missing/invalid")
    spec = document(C / "qualification_spec_18.json", consumed)
    require(spec["arms"] == plan["arms"], "qualification spec/plan differs")
    for name in SOURCE_NAMES:
        in_plan(C / name, plan, consumed)
    for name in ("plan_go_B18.json", "development_readback_18_host_receipt.json",
                 "development_filtered_01/host_receipt.json",
                 "development_filtered_01/worker_receipt.json",
                 "development_filtered_01/session.json"):
        in_plan(C / name, plan, consumed)
    development_scores, dev_ledger = development(plan, consumed)
    readbacks, ledgers, task_gates, initial_pairs = {}, {}, {}, {}
    for arm in ("primary", "mirror"):
        readback, ledger = read_arm(arm, plan, plan_path, consumed)
        readbacks[arm], ledgers[arm] = readback, ledger
        run = C / plan["arms"][arm]["output_directory"]
        gates, pairs = score_table(readback, plan["arms"][arm], run, consumed, paired=True)
        task_gates[arm], initial_pairs[arm] = gates, pairs
    require(tuple(task_gates["primary"]) == CASES
            and tuple(task_gates["mirror"]) == ("flat_1p2_yaw",)
            and readbacks["mirror"]["six_task_zero_pair"] is None,
            "primary/mirror scenario or RL boundary differs")
    rl = contribution(readbacks["primary"]["six_task_zero_pair"],
                      task_gates["primary"], plan["arms"]["primary"])
    totals = sum_ledgers({"development": dev_ledger, **ledgers})
    require(totals["reserved_controls"] == 28000
            and totals["reserved_normal_native"] == 140000
            and totals["compiler_native_returns"] == 6
            and totals["actual_controls"] <= totals["reserved_controls"]
            and totals["actual_normal_native"] == 5*totals["actual_controls"]
            and totals["critic_rows"] == totals["training_calls"] == 0,
            "C18 B+C budget differs")
    old = document(OLD / "qualification_summary_17.json", consumed)
    require(plan["inputs"][str(OLD / "qualification_summary_17.json")]
            == consumed[str(OLD / "qualification_summary_17.json")]
            and old["grouped_required_scene_count"] == 7,
            "C17 comparison identity/shape differs")
    old_actual = old["development_B_plus_C_actual_totals"]
    combined_actual = {
        "actual_controls": old_actual["actual_controls"]+totals["actual_controls"],
        "actual_normal_native": old_actual["actual_normal_native"]+totals["actual_normal_native"],
        "compiler_native_returns": old_actual["compiler_native_returns"]+totals["compiler_native_returns"],
        "policy_load_calls": old_actual["policy_load_calls"]+totals["policy_load_calls"],
        "torch_load_calls": old_actual["torch_load_calls"]+totals["torch_load_calls"],
        "predict_api_calls": old_actual["predict_api_calls"]+totals["predict_api_calls"],
        "actor_rows": old_actual["actor_rows"]+totals["actor_rows"],
        "critic_rows": old_actual["critic_rows"]+totals["critic_rows"],
    }
    require(combined_actual["actual_controls"] <= 60600
            and combined_actual["actual_normal_native"] <= 303000
            and combined_actual["compiler_native_returns"] <= 16,
            "C17+C18 total exceeds registered ceiling")
    old_gates = old["task_gates"]
    differences = {arm: {case: {actor: {
        "C17_task_passed": old_gates[arm][case][actor]["task_passed"],
        "C17_task_reasons": old_gates[arm][case][actor]["task_reasons"],
        "C18_task_passed": task_gates[arm][case][actor]["task_passed"],
        "C18_task_reasons": task_gates[arm][case][actor]["task_reasons"],
        "C17_hold_mean_com_vx_mps": old_gates[arm][case][actor]["hold_mean_com_vx_mps"],
        "C18_hold_mean_com_vx_mps": task_gates[arm][case][actor]["hold_mean_com_vx_mps"],
        "C17_hold_vx_rms_mps": old_gates[arm][case][actor]["hold_vx_rms_mps"],
        "C18_hold_vx_rms_mps": task_gates[arm][case][actor]["hold_vx_rms_mps"],
    } for actor in actors} for case, actors in task_gates[arm].items()}
        for arm in ("primary", "mirror")}
    grouped = [task_gates["primary"][case]["grouped_continue"]["task_passed"]
               for case in CASES] + [task_gates["mirror"]["flat_1p2_yaw"][
                   "grouped_continue"]["task_passed"]]
    zero = [task_gates["primary"][case]["zero"]["task_passed"] for case in CASES]
    zero.append(task_gates["mirror"]["flat_1p2_yaw"]["zero"]["task_passed"])
    result = {"schema": "d1-control-repair-seven-scene-qualification-summary-18-v1",
              "qualification_passed": all(grouped),
              "grouped_passed_scene_count": sum(grouped),
              "grouped_required_scene_count": 7,
              "zero_passed_scene_count": sum(zero), "zero_recorded_scene_count": 7,
              "development_task_gates": development_scores,
              "development_precondition_passed": all(
                  row["grouped_continue"]["task_passed"]
                  for row in development_scores.values()),
              "task_gates": task_gates,
              "within_worker_zero_grouped_initial_state": initial_pairs,
              "development_vs_C17_initial_state": plan["development_initial_state_pairs"],
              "primary_six_task_RL_contribution": rl,
              "worker_ledgers": {"development": dev_ledger, **ledgers},
              "C18_B_plus_C_actual_totals": totals,
              "C17_B_plus_C_actual_totals": old_actual,
              "C17_plus_C18_actual_totals": combined_actual,
              "C17_vs_C18_task_differences": differences,
              "plan_identity": consumed[str(plan_path)],
              "source_closure": "All consumed inputs hashed; planned sources matched actual plan inputs; independent readers verified the full saved source/physics closure.",
              "interpretation_limit": "Seven deterministic fixed oracle scripts; seed labels are scenario identifiers, not independent physical randomization. Baseline controller repair is separate from RL contribution.",
              "not_qualified": ["default GUI", "free keyboard driving", "hardware",
                                "lateral motion", "jump", "15 mm step", "self recovery"]}
    result["input_sha256"] = dict(sorted(consumed.items()))
    result["script_sha256"] = digest(Path(__file__))
    require(not ({name.partition(".")[0] for name in sys.modules} & FORBIDDEN),
            "forbidden model/physics module loaded")
    result["execution"] = {"wall_elapsed_s": time.monotonic()-started,
                           "model_runs": 0, "physics_steps": 0}
    rows = csv_rows(development_scores, task_gates, old_gates)
    fields = list(rows[0])
    # Prepare both serializations before creating either exclusive output.
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with SUMMARY.open("x", encoding="utf-8") as stream:
        stream.write(text)
    with TABLE.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
