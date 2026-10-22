"""Candidate guards and gate semantics; no physics steps or model construction."""

import json
import signal
import sys
from dataclasses import replace

import numpy as np
import pytest

from scripts import estimator_candidate49 as candidate49
from scripts.estimator_candidate49 import (
    CONTACT_FIELDS,
    PREREG,
    apply_candidate,
    evaluate_gate,
    experiment_cases,
    packets_from_archive,
    write_json,
)
from wheel_legged_control.d1.model import NOMINAL_JOINT_POSITION
from wheel_legged_control.d1.state_estimation import D1StateEstimate


def snapshot():
    return D1StateEstimate(
        sequence=2, control_time_s=0.02, measurement_time_s=0.0,
        base_position=np.array([0.0, 0.0, 0.48]), base_rotation=np.eye(3),
        base_linear_velocity_body=np.array([0.3, 0.0, 0.0]),
        base_angular_velocity_body=np.array([0.0, 0.1, 0.0]),
        base_linear_velocity_world=np.array([0.3, 0.0, 0.0]),
        base_angular_velocity_world=np.array([0.0, 0.1, 0.0]),
        joint_position=NOMINAL_JOINT_POSITION, joint_velocity=np.zeros(16),
        foot_position=np.ones((4, 3)), foot_jacobian=np.ones((4, 3, 4)),
        wheel_contact=np.ones(4, dtype=bool), wheel_contact_point=np.ones((4, 3)),
        wheel_contact_normal=np.ones((4, 3)), wheel_contact_jacobian=np.ones((4, 3, 4)),
        undesired_ground_contacts=0,
    )


def synthetic_results(phase):
    return [{
        **case, "steps": 400, "termination_reason": "time_limit",
        "tracking_rmse_mps": 0.1 if case["arm"] == "none" else 0.09,
        "replay_bitwise_equal": True, "provenance_invariants_pass": True,
        "raw_pitch_rmse_rad": 0.02, "candidate_pitch_rmse_rad": 0.019,
        "archive_replay_max_abs_difference": 0.0,
    } for case in experiment_cases(phase)]


def test_postfusion_prediction_keeps_age_contacts_and_raw_snapshot():
    raw = snapshot()
    assert apply_candidate(raw, "none").control_state is raw
    result = apply_candidate(raw, "constant_velocity")
    assert result.status == "applied"
    assert result.control_state.age_s == raw.age_s == 0.02
    assert result.control_state.base_rpy[1] == pytest.approx(0.002)
    assert result.control_state.base_position[0] > raw.base_position[0]
    np.testing.assert_array_equal(raw.base_position, [0, 0, 0.48])
    for name in CONTACT_FIELDS:
        np.testing.assert_array_equal(getattr(result.control_state, name), getattr(raw, name))


def test_existing_rejection_thresholds_and_zero_delay_are_preserved():
    raw = snapshot()
    no_delay = replace(raw, measurement_time_s=raw.control_time_s)
    assert apply_candidate(no_delay, "constant_velocity").control_state is no_delay
    too_old = replace(raw, control_time_s=0.051)
    assert apply_candidate(too_old, "constant_velocity").status == "horizon_exceeded"
    fast_leg = replace(raw, joint_velocity=np.full(16, 18.0))
    result = apply_candidate(fast_leg, "constant_velocity")
    assert result.status == "kinematic_horizon_exceeded"
    assert result.control_state is fast_leg


def test_offline_gate_uses_all_36_including_failed_prefixes():
    rows = synthetic_results("offline")
    assert len(rows) == 36
    rows[0]["archive_termination_reason"] = "fallen"
    assert evaluate_gate("offline", rows)["passed"]
    assert not evaluate_gate("offline", rows[1:])["passed"]
    rows[0]["archive_replay_max_abs_difference"] = 1e-9
    assert not evaluate_gate("offline", rows)["passed"]


def test_offline_gate_blocks_terrain_regression_despite_pooled_improvement():
    rows = synthetic_results("offline")
    for row in rows:
        row["candidate_pitch_rmse_rad"] = 0.022 if row["terrain"] == "flat" else 0.01
    gate = evaluate_gate("offline", rows)
    assert gate["candidate_over_raw_pitch_ratio"] < 0.98
    assert not gate["passed"]


def test_preflight_rejects_new_failure_and_missing_common_completions():
    rows = synthetic_results("preflight")
    assert len(rows) == 12
    assert evaluate_gate("preflight", rows)["passed"]
    rows[1].update(termination_reason="fallen", steps=300)
    assert not evaluate_gate("preflight", rows)["passed"]
    for row in rows:
        row.update(termination_reason="fallen", steps=300)
    gate = evaluate_gate("preflight", rows)
    assert not gate["passed"]
    assert gate["common_complete_pairs"] == 0
    assert len(gate["failed_cases"]) == 12


def test_full_requires_six_extra_delayed_completions_and_zero_delay_safety():
    rows = synthetic_results("full")
    assert len(rows) == 144
    assert not evaluate_gate("full", rows)["passed"]  # Gain is zero, despite lower RMSE.
    baseline_delayed = [row for row in rows if row["arm"] == "none" and row["delay_steps"] == 2]
    for row in baseline_delayed[:6]:
        row.update(termination_reason="fallen", steps=300)
    assert evaluate_gate("full", rows)["passed"]
    rows[1].update(termination_reason="fallen", steps=300)  # Zero-delay candidate.
    assert not evaluate_gate("full", rows)["passed"]


def test_preflight_does_not_replace_out_of_bounds_failure_with_new_fall():
    rows = synthetic_results("preflight")
    rows[0].update(termination_reason="out_of_bounds", steps=300)
    rows[1].update(termination_reason="fallen", steps=300)
    gate = evaluate_gate("preflight", rows)
    assert gate["delay20_completion_gain"] == 0
    assert gate["common_tracking_rmse_ratio"] < 0.98
    assert not gate["no_extra_falls"]
    assert not gate["passed"]


def test_full_rejects_delay_regression_that_pooled_rmse_would_hide():
    rows = synthetic_results("full")
    baseline_delayed = [row for row in rows if row["arm"] == "none" and row["delay_steps"] == 2]
    for row in baseline_delayed[:6]:
        row.update(termination_reason="fallen", steps=300)
    for row in rows:
        if row["arm"] == "constant_velocity":
            row["tracking_rmse_mps"] = 0.106 if row["delay_steps"] == 2 else 0.099
    gate = evaluate_gate("full", rows)
    assert gate["common_tracking_rmse_ratio"] < 1.05
    assert gate["common_tracking_rmse_ratios_by_delay"]["2"] > 1.05
    assert not gate["passed"]


def test_partial_results_are_never_overwritten_and_truth_cannot_enter_packets(tmp_path):
    target = tmp_path / "result.json"
    write_json(target, {"status": "partial"})
    with pytest.raises(FileExistsError):
        write_json(target, {"status": "complete"})
    assert json.loads(target.read_text())["status"] == "partial"
    packet_path = tmp_path / "measurements.npz"
    np.savez(packet_path, base_position=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="non-sensor"):
        packets_from_archive(packet_path)


def test_preregistered_budgets_match_executable_matrix():
    prereg = json.loads(PREREG.read_text())
    for phase in ("offline", "preflight", "full"):
        assert len(experiment_cases(phase)) == prereg[phase]["cases"]
    assert prereg["sole_variable"] == {"latency_compensation": ["none", "constant_velocity"]}


def test_sigterm_retains_active_case_and_marks_physics_counts_unknown(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    write_json(archive / "summary.json", {"cases": []})
    registration = tmp_path / "preregistered.json"
    write_json(registration, {"source_sha256": {}, "archive": str(archive)})
    monkeypatch.setattr(candidate49, "PREREG", registration)
    previous = tmp_path / "previous.json"
    write_json(previous, {
        "phase": "offline", "status": "complete", "gate": {"passed": True},
        "preregistration_sha256": candidate49.sha256(registration),
        "script_sha256": candidate49.sha256(candidate49.Path(candidate49.__file__)),
    })
    output = tmp_path / "new_attempt"
    monkeypatch.setattr(sys, "argv", ["candidate49", "--phase", "preflight", "--output",
                                    str(output), "--previous-summary", str(previous)])

    def interrupted_case(case, destination):
        (destination / f"{candidate49.case_name(case)}.csv").write_text("step\n1\n2\n")
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(candidate49, "run_case", interrupted_case)
    old_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError):
        candidate49.main()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["active_case"] == experiment_cases("preflight")[0]
    assert failure["active_case_logged_control_prefix"] == 2
    assert failure["control_step_lower_bound"] == 2
    assert failure["executed_control_steps"] is None
    assert failure["executed_native_steps"] is None
    assert not failure["counts_exact"]
    assert not failure["resume_permitted"]
    assert signal.getsignal(signal.SIGTERM) == old_handler
    assert not (output / "summary.json").exists()
