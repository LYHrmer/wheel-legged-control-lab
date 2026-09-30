"""Synthetic final-eligibility decisions; no replay or simulation."""
from copy import deepcopy
from decision22 import decide


def reports():
    valid = {"training_valid": True, "coverage_valid": True}
    ratio = lambda n: {"numerator": n, "denominator": 1.}
    pool = {"complete": True, "actors": {"B": {"all_tasks_passed": True}},
            "comparisons": {"B/A": {"error": ratio(.85), "cost": ratio(1.)},
                            "B/old": {"error": ratio(.90), "cost": ratio(1.)},
                            "B/zero": {"error": ratio(1.), "cost": ratio(1.1)}},
            "family_B_over_A": {f: {m: ratio(1.) for m in
                ("drive_error_mean", "hold_error_mean")} for f in ("yaw", "bumps")}}
    dev = {"equal_case_pool": pool, "original_six_pool": deepcopy(pool),
           "source_closure_verified": True, "regression_reuse": {"complete": True},
           "A_reuse": {"passed": True}, "enabled_arms": ["A", "B"],
           "floors": {a: {"numeric_gate_passed": True} for a in ("A", "B")},
           "heldout_summaries": [{} for _ in range(30)],
           "numeric_scores": [{"experiment_actor": "B", "case_id": n, "task_passed": True}
                for n in ("flat_0p6", "flat_1p6", "flat_1p2_yaw", "bumps_0p4",
                          "rough_0p35", "ramp_0p45_complete", "flat_1p2_yaw_mirror")]}
    return valid, deepcopy(valid), dev, {"passed": True}


def test_performance_and_coverage_do_not_select_final_pair():
    args = reports()
    args[1]["coverage_valid"] = False
    args[2]["equal_case_pool"]["complete"] = False
    args[2]["original_six_pool"]["complete"] = False
    for row in args[2]["numeric_scores"]:
        row["task_passed"] = False
    result = decide(*args)
    assert result["final_execution_allowed"]
    assert not result["development_descriptive_gates_passed"]
    assert not result["continue_training"] and result["candidate_stage"] == 1


def test_missing_records_failed_floor_or_invalid_training_block_final():
    for fault in ("missing_record", "failed_floor", "invalid_training", "invalid_reuse", "unpaired"):
        args = reports()
        if fault == "missing_record": args[2]["heldout_summaries"].pop()
        elif fault == "failed_floor": args[2]["floors"]["B"]["numeric_gate_passed"] = False
        elif fault == "invalid_training": args[1]["training_valid"] = False
        elif fault == "invalid_reuse": args[2]["A_reuse"]["passed"] = False
        else: args[3]["passed"] = False
        assert not decide(*args)["final_execution_allowed"], fault


def test_all_gates_pass_never_authorizes_extension():
    result = decide(*reports())
    assert result["final_execution_allowed"] and result["development_descriptive_gates_passed"]
    assert result["continue_training"] is False and result["candidate_stage"] == 1
