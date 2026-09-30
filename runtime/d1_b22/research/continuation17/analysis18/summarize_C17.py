"""Pure qualification summary from C17's two independent saved readbacks.

No controller, model, training, or physics modules are imported. Readback
failure, missing paired evidence, or source mismatch fails closed. This reports
seven deterministic fixed command scripts, not independent random seeds.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
from pathlib import Path
import sys
import time


_IMPORT = builtins.__import__
_FORBIDDEN = frozenset(("torch", "mujoco", "stable_baselines3", "gym", "gymnasium",
                       "wheel_legged_control", "engine_binding"))


def _guard(name, *args, **kwargs):
    if name.partition(".")[0] in _FORBIDDEN:
        raise RuntimeError("model/physics import denied: " + name)
    return _IMPORT(name, *args, **kwargs)


builtins.__import__ = _guard
import numpy as np


C = Path(__file__).resolve().parent.parent
OUTPUT = C / "qualification_summary_17.json"
PLAN = C / "plan_go_C17.json"
PLAN_SHA256 = "7bcc906b572f650f088bfe2086199db8e9cfda4189d4300030f191b480f50201"
CASES = ("flat_0p6", "flat_1p6", "flat_1p2_yaw", "bumps_0p4",
         "rough_0p35", "ramp_0p45_complete")
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
CRITICAL = (
    "qualification_contract_17.md", "qualification_spec_17.json", "go_C17.md",
    "development_summary_17.json", "development_baseline_readback_17.json",
    "development_yaw_readback_17.json", "development_ramp_readback_17.json",
    "controller17.py", "residual17.py", "verify_control17.py", "read_eval17.py",
    "score17.py", "recipes17.py", "host17.py", "worker17.py", "eval17.py",
    "floor_bridge17.py", "offline_floor17.py", "readback_host17_v02.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def file(path: Path, sources: dict[str, dict]) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or linked input: " + str(path))
    path = path.resolve(strict=True)
    sources[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return path


def doc(path: Path, sources: dict[str, dict]) -> dict:
    item = json.loads(file(path, sources).read_text())
    if not isinstance(item, dict):
        raise ValueError("JSON input is not an object: " + str(path))
    return item


def matches_plan(path: Path, plan: dict, sources: dict[str, dict]) -> bool:
    item = file(path, sources)
    expected = plan["inputs"].get(str(item))
    if expected != sources[str(item)]:
        raise ValueError("C frozen input SHA/size differs: " + str(item))
    return True


def bitwise_pair(first: Path, second: Path, sources: dict[str, dict]) -> dict:
    with np.load(file(first, sources), allow_pickle=False) as a, np.load(
            file(second, sources), allow_pickle=False) as b:
        fields = {name: bool(a[name].shape == b[name].shape
                             and a[name].dtype == b[name].dtype
                             and a[name].tobytes(order="C") == b[name].tobytes(order="C"))
                  for name in PAIR_FIELDS}
    return {"all_five_bitwise_equal": all(fields.values()), "fields": fields}


def gate(score: dict) -> dict:
    """Retain every task gate and full failure reasons without numeric traces."""
    required = ("record", "safety", "speed", "yaw", "final_window", "stopping",
                "terrain_channel", "ramp_geometry", "rl_terms", "task_reasons")
    if any(name not in score for name in required):
        raise ValueError("saved score omitted a required gate")
    if type(score.get("task_passed")) is not bool or not isinstance(
            score.get("task_reasons"), list):
        raise ValueError("saved task gate has invalid pass/reason fields")
    result = {"task_passed": score["task_passed"],
              "task_reasons": score["task_reasons"],
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
        geom = score["ramp_geometry"]
        result["ramp_positive_wheel_load_by_geom"] = geom[
            "positive_wheel_load_native_counts_by_geom"]
        result["wheel_collision_world_x_min_m_by_index"] = geom[
            "wheel_collision_world_x_min_m_by_index"]
        result["all_ramp_world_x_max_m"] = geom["all_ramp_world_x_max_m"]
    return result


def run_receipt(arm: str, readback_path: Path, plan: dict,
                sources: dict[str, dict]) -> tuple[dict, dict, dict]:
    spec = plan["arms"][arm]
    run = C / spec["output_directory"]
    readback = doc(readback_path, sources)
    reader_host = doc(readback_path.with_name(readback_path.stem + "_host_receipt.json"), sources)
    host = doc(run / "host_receipt.json", sources)
    worker = doc(run / "worker_receipt.json", sources)
    session = doc(run / "session.json", sources)
    if (readback.get("schema") != "d1-world-upright-stage17-independent-eval-readback-v1"
            or readback.get("run") != str(run.resolve())
            or readback.get("controller_variant") != "combined"
            or readback.get("source_closure_verified") is not True
            or readback.get("checkpoint_ZIP_verified") is not True
            or readback.get("physical_calls_performed_by_reader") != 0
            or readback.get("model_loaded_by_reader") is not False
            or reader_host.get("failure") is not None
            or reader_host.get("exit_code") != 0
            or reader_host.get("elapsed_s", 301) > 300
            or reader_host.get("controls") != 0
            or reader_host.get("model_calls") != 0
            or reader_host.get("output_identity") != sources[str(readback_path.resolve())]
            or reader_host.get("reader_identity") != plan["inputs"][str(C / "read_eval17.py")]
            or host.get("failure") is not None
            or host.get("exit_code") != 0
            or host.get("cleanup_errors") != []
            or host.get("changed_sources") != []
            or host.get("postcheck_complete") is not True
            or host.get("no_live_owned_processes") is not True
            or host.get("fully_reserved_budget_closed") is not True
            or worker.get("schema") != "d1-control-repair-worker-17-v1"
            or worker.get("arm") != arm
            or worker.get("failure") is not None
            or worker.get("execution_complete") is not True
            or worker.get("cleanup_errors") != []
            or worker.get("warnings") != []
            or session.get("plan_identity") != sources[str(PLAN)]
            or session.get("source_hashes", {}).get(str(PLAN)) != sources[str(PLAN)]
            or session.get("case_specs") != spec["case_specs"]
            or session.get("floor_seed") != spec["floor_seed"]
            or session.get("controller_variant") != "combined"):
        raise ValueError("C worker/independent readback/source closure differs: " + arm)
    expected_controls = spec["controls"]
    actual_controls = readback["actual_completed_controls"]
    if (actual_controls > expected_controls
            or readback["actual_normal_native_returns"] != 5*actual_controls
            or readback["actual_compiler_native_returns"] != 2
            or worker["result"]["completed_controls"] != actual_controls
            or worker["C_final"]["construction_attempts"] != 2
            or worker["C_final"]["construction_returns"] != 2
            or worker["C_final"]["control_attempts"] != 5*actual_controls
            or worker["C_final"]["control_returns"] != 5*actual_controls
            or worker["python"]["control_completed"] != actual_controls
            or worker["python"]["native_returned"] != 5*actual_controls
            or worker["native_guard"]["native_checked"] != 5*actual_controls):
        raise ValueError("C model/native/control ledger differs: " + arm)
    calls = worker["model_calls"]["counts"]
    expected_predict = 1 + readback["floor"]["completed_controls"] + sum(
        row["predict_calls"] for row in readback["heldout_summaries"])
    expected_calls = {"load": 1, "torch_load": 3, "predict": expected_predict,
                      "save": 0, "learn": 0, "train": 0, "forward": 0,
                      "evaluate_actions": 0, "predict_values": 0, "backward": 0}
    if any(calls.get(name, {}).get("attempted", 0) != value
           or calls.get(name, {}).get("returned", 0) != value
           for name, value in expected_calls.items()):
        raise ValueError("C model API budget differs: " + arm)
    expected_actor_rows = expected_predict + 31  # 32-row probe, one row per later call.
    predict = calls["predict"]
    if (predict["rows_attempted"] != expected_actor_rows
            or predict["rows_returned"] != expected_actor_rows
            or predict["phases"]["probe"]["rows"] != 32
            or worker["model_calls"]["actor_rows"] != expected_actor_rows
            or worker["model_calls"]["critic_rows"] != 0):
        raise ValueError("C model API actor/critic row ledger differs: " + arm)
    ledger = {"reserved_controls": expected_controls, "actual_controls": actual_controls,
              "reserved_normal_native": 5*expected_controls,
              "actual_normal_native": 5*actual_controls,
              "compiler_native_returns": 2, "model_calls": expected_calls,
              "actor_rows": expected_actor_rows, "critic_rows": 0,
              "worker_closed": True, "host_closed": True, "independent_reader_closed": True}
    return readback, session, ledger


def development_budget(plan: dict, sources: dict[str, dict]) -> dict:
    """Count B from its three frozen worker receipts, not remembered totals."""
    workers = {}
    for arm in ("baseline", "yaw", "ramp"):
        path = C / f"development_{arm}_01" / "worker_receipt.json"
        matches_plan(path, plan, sources)
        worker = doc(path, sources)
        if (worker.get("schema") != "d1-control-repair-worker-17-v1"
                or worker.get("arm") != arm
                or worker.get("failure") is not None
                or worker.get("execution_complete") is not True
                or worker.get("cleanup_errors") != []
                or worker.get("warnings") != []):
            raise ValueError("B frozen worker receipt no longer closed: " + arm)
        actual = worker["result"]["completed_controls"]
        calls = worker["model_calls"]["counts"]
        predict = calls["predict"]
        actor_rows = worker["model_calls"]["actor_rows"]
        if (worker["C_final"]["control_returns"] != 5*actual
                or worker["C_final"]["construction_returns"] != 2
                or worker["python"]["control_completed"] != actual
                or worker["native_guard"]["native_checked"] != 5*actual
                or calls["load"]["returned"] != 1
                or calls["torch_load"]["returned"] != 3
                or predict["attempted"] != predict["returned"]
                or predict["rows_attempted"] != actor_rows
                or predict["rows_returned"] != actor_rows
                or predict["phases"]["probe"]["rows"] != 32
                or actor_rows != predict["returned"] + 31
                or worker["model_calls"]["critic_rows"] != 0):
            raise ValueError("B frozen model/native row ledger differs: " + arm)
        workers[arm] = {"controls": actual, "normal_native_returns": 5*actual,
                        "compiler_native_returns": 2,
                        "policy_load_calls": calls["load"]["returned"],
                        "torch_load_calls": calls["torch_load"]["returned"],
                        "predict_api_calls": predict["returned"],
                        "actor_rows": actor_rows, "critic_rows": 0}
    fields = next(iter(workers.values())).keys()
    return {"workers": workers,
            "totals": {field: sum(worker[field] for worker in workers.values())
                       for field in fields}}


def summarize_scores(readback: dict, spec: dict, run: Path,
                     sources: dict[str, dict]) -> tuple[dict, dict]:
    scores = readback["numeric_scores"]
    expected = {(row["case_id"], actor) for row in spec["case_specs"]
                for actor in row["actors"]}
    found = {(row["case_id"], row["experiment_actor"]) for row in scores}
    if len(scores) != len(expected) or found != expected:
        raise ValueError("C readback missing one or more zero/grouped score")
    table, pair_checks = {}, {}
    for item in spec["case_specs"]:
        case_id = item["case_id"]
        table[case_id] = {}
        if item["actors"] != ["zero", "grouped_continue"]:
            raise ValueError("C scenario lost the zero/grouped pairing")
        for actor in item["actors"]:
            candidates = [row for row in scores if row["case_id"] == case_id
                          and row["experiment_actor"] == actor]
            if (len(candidates) != 1
                    or candidates[0]["seed"] != item["seed"]
                    or candidates[0]["mirror"] is not item["mirror"]):
                raise ValueError("C scored task seed/mirror/actor differs")
            table[case_id][actor] = gate(candidates[0])
        folder = run / "heldout"
        pair = bitwise_pair(folder / f"{case_id}_zero" / "initial_state.npz",
                            folder / f"{case_id}_grouped_continue" / "initial_state.npz",
                            sources)
        if not pair["all_five_bitwise_equal"]:
            raise ValueError("C zero/grouped initial state differs: " + case_id)
        pair_checks[case_id] = pair
    return table, pair_checks


def compact_contribution(pair: dict) -> dict:
    if (pair.get("schema") != "d1-course-world-upright-rl16-heldout-pure-pair-17-v1"
            or pair.get("pair_count") != 6 or len(pair.get("pairs", [])) != 6):
        raise ValueError("primary six-task RL contribution block missing")
    return {"RL_contribution_passed": pair["RL_contribution_passed"],
            "all_six_policy_task_qualified": pair["all_six_policy_task_qualified"],
            "prerequisites": pair["prerequisites"],
            "cost_gate": pair["cost_gate"],
            "branch_a_task_conversion": pair["branch_a_task_conversion"],
            "branch_b_sse_improvement": pair["branch_b_sse_improvement"],
            "per_task": [{key: row[key] for key in (
                "case_id", "seed", "zero_task_passed", "policy_task_passed",
                "capability_regression", "task_conversion", "common_task_passing",
                "sse_zero", "sse_policy", "sse_ratio", "sse_no_worse_within_1p02",
                "common_cost_sum_zero", "common_cost_sum_policy")}
                for row in pair["pairs"]],
            "torque_cost_is_normalized_squared_proxy_not_energy": True}


def bind_contribution(pair: dict, scores: dict, spec: dict) -> None:
    """Check the reported six-task arithmetic against the saved case scores."""
    expected_cases = [row["case_id"] for row in spec["case_specs"]]
    if [row["case_id"] for row in pair["pairs"]] != expected_cases:
        raise ValueError("RL contribution pair order/case differs")
    for row, case in zip(pair["pairs"], spec["case_specs"]):
        zero = scores[case["case_id"]]["zero"]
        policy = scores[case["case_id"]]["grouped_continue"]
        if (row["seed"] != case["seed"]
                or row["zero_task_passed"] is not zero["task_passed"]
                or row["policy_task_passed"] is not policy["task_passed"]
                or row["sse_zero"] != zero["drive_sse"]
                or row["sse_policy"] != policy["drive_sse"]):
            raise ValueError("RL contribution arithmetic detached from scored pair")


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-readback", type=Path,
                        default=C / "qualification_primary_readback_17.json")
    parser.add_argument("--mirror-readback", type=Path,
                        default=C / "qualification_mirror_readback_17.json")
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sources: dict[str, dict] = {}
    plan = doc(PLAN, sources)
    if (sources[str(PLAN)]["sha256"] != PLAN_SHA256
            or plan.get("status") != "GO"
            or plan.get("development_precondition_verified") is not True
            or set(plan.get("arms", {})) != {"primary", "mirror"}):
        raise ValueError("C frozen GO plan identity differs")
    spec = doc(C / "qualification_spec_17.json", sources)
    if spec["arms"] != plan["arms"]:
        raise ValueError("C frozen spec/plan scenario table differs")
    for name in CRITICAL:
        matches_plan(C / name, plan, sources)
    before_after = doc(C / "development_summary_17.json", sources)
    if (before_after.get("schema") != "d1-control-repair-development-B-summary-17-v1"
            or before_after.get("source_closure_verified") is not True
            or not all(value.get("all_five_equal") is True for value in
                       before_after["cross_worker_initial_state"].values())):
        raise ValueError("B before/after prerequisite differs")
    b_budget = development_budget(plan, sources)
    readbacks, ledgers, task_gates, paired_initials = {}, {}, {}, {}
    for arm, path in (("primary", args.primary_readback),
                      ("mirror", args.mirror_readback)):
        readback, _session, ledger = run_receipt(arm, path, plan, sources)
        readbacks[arm], ledgers[arm] = readback, ledger
        run = C / plan["arms"][arm]["output_directory"]
        gates, pairs = summarize_scores(readback, plan["arms"][arm], run, sources)
        task_gates[arm], paired_initials[arm] = gates, pairs
        if readback["floor"]["numeric_gate_passed"] is not True:
            raise ValueError("C floor numeric gate failed: " + arm)
    if (tuple(task_gates["primary"]) != CASES
            or tuple(task_gates["mirror"]) != ("flat_1p2_yaw",)
            or readbacks["mirror"]["six_task_zero_pair"] is not None):
        raise ValueError("C seven-scene order or mirror contribution boundary differs")
    contribution = compact_contribution(readbacks["primary"]["six_task_zero_pair"])
    bind_contribution(readbacks["primary"]["six_task_zero_pair"],
                      task_gates["primary"], plan["arms"]["primary"])
    grouped_passes = [task_gates["primary"][name]["grouped_continue"]["task_passed"]
                      for name in CASES]
    grouped_passes.append(task_gates["mirror"]["flat_1p2_yaw"][
        "grouped_continue"]["task_passed"])
    zero_passes = [task_gates["primary"][name]["zero"]["task_passed"]
                   for name in CASES]
    zero_passes.append(task_gates["mirror"]["flat_1p2_yaw"]["zero"]["task_passed"])
    totals = {"reserved_controls": sum(row["reserved_controls"] for row in ledgers.values()),
              "actual_controls": sum(row["actual_controls"] for row in ledgers.values()),
              "reserved_normal_native": sum(row["reserved_normal_native"] for row in ledgers.values()),
              "actual_normal_native": sum(row["actual_normal_native"] for row in ledgers.values()),
              "compiler_native_returns": sum(row["compiler_native_returns"] for row in ledgers.values()),
              "policy_load_calls": 2, "torch_load_calls": 6,
              "predict_api_calls": sum(row["model_calls"]["predict"] for row in ledgers.values()),
              "actor_rows": sum(row["actor_rows"] for row in ledgers.values()),
              "critic_rows": sum(row["critic_rows"] for row in ledgers.values()),
              "training_calls": 0}
    if (totals["reserved_controls"] != 24000
            or totals["reserved_normal_native"] != 120000
            or totals["compiler_native_returns"] != 4
            or totals["predict_api_calls"] > 12602
            or totals["actor_rows"] > 12664
            or totals["actor_rows"] != totals["predict_api_calls"] + 62
            or totals["critic_rows"] != 0):
        raise ValueError("C total reserved/model/native budget differs")
    b_totals = b_budget["totals"]
    b_plus_c = {"actual_controls": b_totals["controls"] + totals["actual_controls"],
                "actual_normal_native": b_totals["normal_native_returns"]
                + totals["actual_normal_native"],
                "compiler_native_returns": b_totals["compiler_native_returns"]
                + totals["compiler_native_returns"],
                "policy_load_calls": b_totals["policy_load_calls"]
                + totals["policy_load_calls"],
                "torch_load_calls": b_totals["torch_load_calls"]
                + totals["torch_load_calls"],
                "predict_api_calls": b_totals["predict_api_calls"]
                + totals["predict_api_calls"],
                "actor_rows": b_totals["actor_rows"] + totals["actor_rows"],
                "critic_rows": b_totals["critic_rows"] + totals["critic_rows"]}
    result = {"schema": "d1-control-repair-seven-scene-qualification-summary-17-v1",
              "qualification_passed": all(grouped_passes),
              "grouped_passed_scene_count": sum(grouped_passes),
              "grouped_required_scene_count": 7,
              "zero_passed_scene_count": sum(zero_passes),
              "zero_recorded_scene_count": 7,
              "task_gates": task_gates,
              "within_worker_zero_grouped_initial_state": paired_initials,
              "development_B_before_after_task_gates": before_after["task_gates"],
              "development_B_cross_worker_initial_state": before_after["cross_worker_initial_state"],
              "primary_six_task_RL_contribution": contribution,
              "worker_ledgers": ledgers, "total_budget": totals,
              "development_B_worker_budgets": b_budget,
              "development_B_plus_C_actual_totals": b_plus_c,
              "frozen_plan_sha256": PLAN_SHA256,
              "frozen_critical_inputs_verified": list(CRITICAL),
              "interpretation_limit": "Seven deterministic fixed oracle simulation scripts; changed seed labels do not constitute independent physical randomization or statistical reliability. Baseline control repair is not RL contribution.",
              "not_qualified": ["default GUI", "free keyboard driving", "hardware",
                                "lateral motion", "jump", "15 mm step", "self recovery"]}
    result["input_sha256"] = dict(sorted(sources.items()))
    result["script_sha256"] = digest(Path(__file__))
    if {name.partition(".")[0] for name in sys.modules} & _FORBIDDEN:
        raise RuntimeError("forbidden model/physics module loaded")
    result["execution"] = {"wall_elapsed_s": time.monotonic()-started,
                           "model_runs": 0, "physics_steps": 0}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
