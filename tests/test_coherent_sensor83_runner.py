"""Pure gate, binding and output-discipline tests; no models or physics."""

import csv
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import coherent_sensor83 as runner


def synthetic():
    return [
        {
            **case,
            "steps": 400,
            "termination_reason": "time_limit",
            "tracking_rmse_mps": 0.1 if case["arm"] == runner.ARMS[0] else 0.097,
            "replay_bitwise_equal": True,
            "provenance_invariants_pass": True,
            "coherent_sample_invariants_pass": True,
            "baseline_reproduction": {"passed": True},
        }
        for case in runner.experiment_cases()
    ]


def test_exact_matrix_budget_and_no_full_authorization():
    cases = runner.experiment_cases()
    assert len(cases) == len({runner.case_name(case) for case in cases}) == 12
    assert {case["seed"] for case in cases} == {17, 29, 43}
    assert {case["noise"] for case in cases} == {"noise", "bias"}
    assert {case["delay_steps"] for case in cases} == {2}
    limits = runner.protocol()["limits"]
    assert limits["task_control_steps"] == 4800
    assert limits["identification_auxiliary_plant_steps"] == 511
    assert limits["total_plant_steps"] == 5311
    assert limits["task_explicit_integration_substeps"] == 24000
    assert limits["identification_explicit_integration_substeps"] == 2555
    assert limits["total_explicit_integration_substeps"] == 26555
    assert limits["candidate_sampling_copy_forward_calls"] == 2406
    assert limits["sensor_rne_post_constraint_calls"] == 4812
    assert not runner.protocol()["full_authorized"]


def test_gate_retains_original_predicate_and_explicit_argument_mapping():
    rows = synthetic()
    result = runner.evaluate_gate(rows)
    assert result["passed"] and result["original_gate_passed"]
    assert result["decision"] == "PREFLIGHT_PASS_FULL_NOT_AUTHORIZED"
    assert not result["performance_qualified"] and not result["full_authorized"]
    assert set(result["completions"]["2"]) == set(runner.ARMS)
    rows[1]["tracking_rmse_mps"] = 0.2
    result = runner.evaluate_gate(rows)
    assert not result["passed"] and not result["original_gate_passed"]
    assert result["decision"] == "REJECTED"


def test_missing_duplicate_or_reordered_cases_cannot_pass():
    rows = synthetic()
    for invalid in (rows[:-1], rows + [rows[0]], rows[::-1]):
        assert not runner.evaluate_gate(invalid)["passed"]


def test_failed_prefixes_never_become_full_pairs_and_new_falls_are_not_hidden():
    rows = synthetic()
    rows[0].update(steps=300, termination_reason="out_of_bounds")
    rows[1].update(steps=300, termination_reason="fallen")
    gate = runner.evaluate_gate(rows)
    assert gate["common_complete_pairs"] == 5
    assert not gate["no_extra_falls"] and not gate["passed"]
    assert len(gate["failed_cases"]) == 2
    for row in rows:
        row.update(steps=300, termination_reason="fallen")
    gate = runner.evaluate_gate(rows)
    assert gate["common_complete_pairs"] == 0 and not gate["passed"]


def test_direction_gate_is_not_relaxed_to_just_five_percent_noninferiority():
    rows = synthetic()
    for row in rows[1::2]:
        row["tracking_rmse_mps"] = 0.099
    gate = runner.evaluate_gate(rows)
    assert gate["common_tracking_rmse_ratio"] == pytest.approx(0.99)
    assert not gate["passed"]
    rows[0].update(steps=300, termination_reason="fallen")
    assert runner.evaluate_gate(rows)["passed"]  # One extra completion, with original RMSE guard.
    for row in rows[1::2]:
        row["tracking_rmse_mps"] = 0.106
    assert not runner.evaluate_gate(rows)["passed"]


@pytest.mark.parametrize("field", ["coherent_sample_invariants_pass", "baseline_reproduction"])
def test_new_integrity_checks_cannot_be_bypassed_by_good_tracking(field):
    rows = synthetic()
    rows[0][field] = {"passed": False} if field == "baseline_reproduction" else False
    gate = runner.evaluate_gate(rows)
    assert gate["original_gate_passed"] and not gate["passed"]


def test_baseline_readback_uses_every_original_column_and_core_metric(tmp_path):
    case = runner.experiment_cases()[0]
    stem = runner.case_name({**case, "arm": "none"})
    row = {
        "step": 1,
        "time_s": 0.01,
        "truth_pitch_rad": 0.003,
        "applied_compensation_status": "disabled",
    }
    with (tmp_path / (stem + ".csv")).open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    metrics = {
        "steps": 1,
        "termination_reason": "fallen",
        "tracking_rmse_mps": 0.1,
        "velocity_estimation_rmse_mps": 0.2,
        "compensation_status_counts": {"disabled": 1},
        "native_steps": 5,
    }
    (tmp_path / (stem + ".json")).write_text(json.dumps(metrics))
    assert runner.baseline_reproduction(case, [row], metrics, tmp_path)["passed"]
    altered = {**row, "truth_pitch_rad": 0.003000000001}
    assert not runner.baseline_reproduction(case, [altered], metrics, tmp_path)["passed"]
    assert not runner.baseline_reproduction(case, [], metrics, tmp_path)["passed"]
    result = runner.baseline_reproduction(case, [row], {**metrics, "steps": 400}, tmp_path)
    assert not result["passed"]


def test_plan_rejects_source_changes_and_full_or_modified_budget(monkeypatch):
    monkeypatch.setattr(runner, "source_binding", lambda: {"frozen.py": "aaa"})
    plan = {"protocol": runner.protocol(), "source_sha256": {"frozen.py": "aaa"}}
    runner.verify_plan(plan)
    changed = deepcopy(plan)
    changed["source_sha256"]["frozen.py"] = "bbb"
    with pytest.raises(ValueError, match="source closure"):
        runner.verify_plan(changed)
    for key, value in (("full_authorized", True), ("phase", "full")):
        changed = deepcopy(plan)
        changed["protocol"][key] = value
        with pytest.raises(ValueError, match="fixed preflight protocol"):
            runner.verify_plan(changed)


def test_plan_and_run_output_are_exclusive_before_any_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "source_binding", lambda: {"frozen.py": "aaa"})
    path = tmp_path / "plan.json"
    runner.build_plan(path)
    with pytest.raises(FileExistsError):
        runner.build_plan(path)
    occupied = tmp_path / "old_attempt"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        runner.execute(path, occupied)


def test_failed_admission_preserves_failure_receipt_and_never_starts_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "source_binding", lambda: {"frozen.py": "new"})
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "protocol": runner.protocol(),
                "source_sha256": {"frozen.py": "old"},
            }
        )
    )
    calls = []
    monkeypatch.setattr(runner, "run_case", lambda *args: calls.append(args))
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="source closure"):
        runner.execute(path, output)
    failure = json.loads((output / "failure.json").read_text())
    assert not calls and failure["executed_control_steps"] == 0
    assert failure["counts_exact"] and len(failure["not_run_cases"]) == 12
    assert not failure["resume_permitted"]


def test_inflight_interruption_records_unknown_counts_and_retains_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "verify_plan", lambda plan: None)
    monkeypatch.setattr(runner, "require_cold_identification_cache", dict)
    path = tmp_path / "plan.json"
    path.write_text("{}")

    def interrupted_case(case, output, budget, completed_controls):
        with (output / (runner.case_name(case) + ".csv")).open("x") as handle:
            handle.write("step,time_s\n1,0.01\n")
        raise InterruptedError("simulated supervised stop; no model work")

    monkeypatch.setattr(runner, "run_case", interrupted_case)
    output = tmp_path / "partial"
    with pytest.raises(InterruptedError):
        runner.execute(path, output)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["executed_control_steps"] is None
    assert failure["task_explicit_integration_substeps"] is None
    assert failure["total_plant_steps"] is None
    assert not failure["counts_exact"]
    assert failure["active_case_logged_prefix"] == failure["control_step_lower_bound"] == 1
    assert len(failure["not_run_cases"]) == 11
    assert list(output.glob("*.csv")) and not (output / "summary.json").exists()


def test_cold_identification_auxiliary_steps_cannot_be_hidden_by_warming_cache(monkeypatch):
    cold = {"hits": 0, "misses": 0, "currsize": 0, "maxsize": 8}
    initialized = {**cold, "misses": 1, "currsize": 1}
    monkeypatch.setattr(runner, "identification_cache", lambda: cold)
    assert runner.require_cold_identification_cache() == cold
    counted = runner.identification_accounting(cold, initialized, first_case=True)
    assert counted["auxiliary_plant_steps"] == 511
    assert counted["auxiliary_explicit_integration_substeps"] == 2555
    reused = runner.identification_accounting(
        initialized,
        {**initialized, "hits": 1},
        first_case=False,
    )
    assert reused["auxiliary_plant_steps"] == 0
    monkeypatch.setattr(runner, "identification_cache", lambda: initialized)
    with pytest.raises(ValueError, match="fresh process"):
        runner.require_cold_identification_cache()
    with pytest.raises(ValueError, match="unexpected cold"):
        runner.identification_accounting(initialized, {**initialized, "hits": 1}, first_case=True)


def test_observer_fk_ledger_uses_delayed_packets_and_counts_both_observers():
    packets = [
        SimpleNamespace(wheel_contact=np.array(contact, dtype=bool))
        for contact in (
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        )
    ]
    ledger = runner.observer_fk_accounting(packets, 2)
    assert ledger["updates_each"] == 1 and ledger["snapshots_each"] == 3
    assert ledger["mj_kinematics"] == ledger["mj_comPos"] == 6
    assert ledger["mj_jacBodyCom"] == 24
    assert ledger["contact_mj_jac"] == 2 * (4 + 2 * 2)
