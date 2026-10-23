"""Candidate50 field isolation, gates and failure retention; no model construction."""

import json
import signal
import sys
from dataclasses import fields, replace

import numpy as np
import pytest

from scripts import estimator_candidate50 as experiment
from wheel_legged_control.d1.model import JOINT_POSITION_HIGH, NOMINAL_JOINT_POSITION
from wheel_legged_control.d1.state_estimation import D1StateEstimate


def snapshot():
    return D1StateEstimate(
        sequence=2, control_time_s=0.02, measurement_time_s=0.0,
        base_position=np.array([0.0, 0.0, 0.48]), base_rotation=np.eye(3),
        base_linear_velocity_body=np.array([0.3, 0.0, 0.0]),
        base_angular_velocity_body=np.array([0.0, 0.1, 0.0]),
        base_linear_velocity_world=np.array([0.3, 0.0, 0.0]),
        base_angular_velocity_world=np.array([0.0, 0.1, 0.0]),
        joint_position=NOMINAL_JOINT_POSITION, joint_velocity=np.full(16, 0.2),
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
        "single_field_ablation_bitwise_pass": True, "pitch_bitwise_equal_archived49": True,
        "raw_pitch_rmse_rad": 0.02, "candidate_pitch_rmse_rad": 0.019,
        "archive_replay_max_abs_difference": 0.0,
    } for case in experiment.experiment_cases(phase)]


def test_only_twelve_leg_positions_change_and_the_raw_snapshot_is_not_mutated():
    raw = snapshot()
    before = {field.name: np.asarray(getattr(raw, field.name)).tobytes() for field in fields(raw)}
    cv = experiment.apply_candidate(raw, "constant_velocity")
    candidate = experiment.apply_candidate(raw, experiment.CANDIDATE)
    assert cv.status == candidate.status == "applied"
    changed = [field.name for field in fields(raw) if not experiment.bitwise_equal(
        getattr(cv.control_state, field.name), getattr(candidate.control_state, field.name))]
    assert changed == ["joint_position"]
    indices = np.flatnonzero(cv.control_state.joint_position != candidate.control_state.joint_position)
    np.testing.assert_array_equal(indices, experiment.LEG_INDICES)
    np.testing.assert_array_equal(candidate.control_state.joint_position[experiment.LEG_INDICES],
                                  raw.joint_position[experiment.LEG_INDICES])
    np.testing.assert_array_equal(candidate.control_state.joint_position[experiment.WHEEL_INDICES],
                                  cv.control_state.joint_position[experiment.WHEEL_INDICES])
    assert candidate.control_state.age_s == raw.age_s == 0.02
    assert before == {field.name: np.asarray(getattr(raw, field.name)).tobytes()
                      for field in fields(raw)}
    assert experiment.apply_candidate(raw, "none").control_state is raw


def test_only_the_added_pd_velocity_term_is_removed():
    raw = snapshot()
    cv = experiment.apply_candidate(raw, "constant_velocity").control_state
    candidate = experiment.apply_candidate(raw, experiment.CANDIDATE).control_state
    leg = experiment.LEG_INDICES
    cv_pd = 40.0 * (NOMINAL_JOINT_POSITION[leg] - cv.joint_position[leg])
    cv_pd -= 1.5 * cv.joint_velocity[leg]
    candidate_pd = 40.0 * (NOMINAL_JOINT_POSITION[leg] - candidate.joint_position[leg])
    candidate_pd -= 1.5 * candidate.joint_velocity[leg]
    np.testing.assert_allclose(candidate_pd - cv_pd, 40.0 * raw.age_s * raw.joint_velocity[leg],
                               rtol=1e-12, atol=1e-12)


def test_original_guards_are_kept_even_though_leg_position_will_remain_measured():
    raw = snapshot()
    near_limit = raw.joint_position.copy()
    near_limit[0] = JOINT_POSITION_HIGH[0] - 1e-5
    examples = [
        (replace(raw, control_time_s=0.051), "horizon_exceeded"),
        (replace(raw, joint_velocity=np.full(16, 18.0)), "kinematic_horizon_exceeded"),
        (replace(raw, joint_position=near_limit), "kinematic_horizon_exceeded"),
        (replace(raw, measurement_time_s=raw.control_time_s), "bypassed"),
    ]
    for measured, status in examples:
        cv = experiment.apply_candidate(measured, "constant_velocity")
        candidate = experiment.apply_candidate(measured, experiment.CANDIDATE)
        assert cv.status == candidate.status == status
        assert candidate.control_state is measured
        experiment.assert_single_field_ablation(measured, cv, candidate)
    with pytest.raises(ValueError, match="unregistered arm"):
        experiment.apply_candidate(raw, "half_horizon")


def test_mixed_time_provenance_and_bitwise_field_guard_are_explicit():
    raw = snapshot()
    cv = experiment.apply_candidate(raw, "constant_velocity")
    candidate = experiment.apply_candidate(raw, experiment.CANDIDATE)
    times = experiment.state_times(raw, candidate, experiment.CANDIDATE)
    assert times["leg_joint_position_time_s"] == times["contact_geometry_time_s"] == 0.0
    assert times["predicted_geometry_time_s"] == 0.02
    assert times["mixed_time_control_input"] and not times["whole_state_current"]
    bad_position = candidate.control_state.base_position.copy()
    bad_position[1] = -0.0  # A numeric equality check would miss this unauthorized bit change.
    bad = replace(candidate, control_state=replace(candidate.control_state,
                                                   base_position=bad_position))
    with pytest.raises(AssertionError, match="base_position"):
        experiment.assert_single_field_ablation(raw, cv, bad)


def test_offline_requires_all_cases_field_equality_and_original_pitch_gates():
    rows = synthetic_results("offline")
    assert len(rows) == 36
    rows[0]["archive_termination_reason"] = "fallen"
    assert experiment.evaluate_gate("offline", rows)["passed"]
    assert not experiment.evaluate_gate("offline", rows[1:])["passed"]
    for key in ("single_field_ablation_bitwise_pass", "pitch_bitwise_equal_archived49"):
        rows[0][key] = False
        assert not experiment.evaluate_gate("offline", rows)["passed"]
        rows[0][key] = True
    rows[0]["archive_replay_max_abs_difference"] = 1e-9
    assert not experiment.evaluate_gate("offline", rows)["passed"]
    rows[0]["archive_replay_max_abs_difference"] = 0.0
    for row in rows:
        row["candidate_pitch_rmse_rad"] = 0.022 if row["terrain"] == "flat" else 0.01
    gate = experiment.evaluate_gate("offline", rows)
    assert gate["candidate_over_raw_pitch_ratio"] < 0.98 and not gate["passed"]


def test_preflight_cannot_use_poor_cv_reference_as_a_relaxed_performance_baseline():
    rows = synthetic_results("preflight")
    assert len(rows) == 18 and experiment.evaluate_gate("preflight", rows)["passed"]
    for row in rows:
        if row["arm"] == "constant_velocity":
            row["tracking_rmse_mps"] = 1.0
        elif row["arm"] == experiment.CANDIDATE:
            row["tracking_rmse_mps"] = 0.11
    gate = experiment.evaluate_gate("preflight", rows)
    assert gate["performance_baseline"] == "none"
    assert gate["mechanism_reference"]["candidate_over_cv_tracking_ratio"] < 0.2
    assert gate["common_tracking_rmse_ratio"] > 1.05 and not gate["passed"]
    assert not experiment.evaluate_gate("preflight", rows[:1] + rows[2:])["passed"]


def test_preflight_keeps_fall_and_common_complete_pair_guards():
    rows = synthetic_results("preflight")
    rows[0].update(termination_reason="out_of_bounds", steps=300)
    rows[2].update(termination_reason="fallen", steps=300)
    gate = experiment.evaluate_gate("preflight", rows)
    assert gate["delay20_completion_gain"] == 0
    assert not gate["no_extra_falls"] and not gate["passed"]
    for row in rows:
        row.update(termination_reason="fallen", steps=300)
    gate = experiment.evaluate_gate("preflight", rows)
    assert gate["common_complete_pairs"] == 0 and not gate["passed"]
    assert len(gate["failed_cases"]) == 18


def test_full_keeps_six_extra_completions_zero_delay_safety_and_per_delay_rmse():
    rows = synthetic_results("full")
    assert len(rows) == 144 and not experiment.evaluate_gate("full", rows)["passed"]
    delayed_baseline = [r for r in rows if r["arm"] == "none" and r["delay_steps"] == 2]
    for row in delayed_baseline[:6]:
        row.update(termination_reason="fallen", steps=300)
    assert experiment.evaluate_gate("full", rows)["passed"]
    rows[1].update(termination_reason="fallen", steps=300)
    assert not experiment.evaluate_gate("full", rows)["passed"]
    rows[1].update(termination_reason="time_limit", steps=400)
    for row in rows:
        if row["arm"] == experiment.CANDIDATE:
            row["tracking_rmse_mps"] = 0.106 if row["delay_steps"] == 2 else 0.099
    gate = experiment.evaluate_gate("full", rows)
    assert gate["common_tracking_rmse_ratio"] < 1.05
    assert gate["common_tracking_rmse_ratios_by_delay"]["2"] > 1.05 and not gate["passed"]


def test_registration_budgets_match_all_fresh_execution_matrices():
    prereg = json.loads(experiment.PREREG.read_text())
    for phase in ("offline", "preflight", "full"):
        assert len(experiment.experiment_cases(phase)) == prereg[phase]["cases"]
    assert prereg["preflight"]["control_step_max"] == 7200
    assert prereg["full"]["control_step_max"] == 57600
    assert prereg["sole_variable"]["field"] == "joint_position[12 leg indices]"
    assert experiment.COUNT_BOUNDARY["total_native_calls"] is None


def test_output_cannot_overwrite_partial_and_truth_cannot_enter_measurement_archive(tmp_path):
    target = tmp_path / "result.json"
    experiment.write_json(target, {"status": "partial"})
    with pytest.raises(FileExistsError):
        experiment.write_json(target, {"status": "complete"})
    packet_path = tmp_path / "measurements.npz"
    np.savez(packet_path, base_position=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="non-sensor"):
        experiment.packets_from_archive(packet_path)


def test_previous_phase_requires_unchanged_outputs_and_recomputed_gate(tmp_path):
    artifact = tmp_path / "artifact.json"
    experiment.write_json(artifact, {"audit": True})
    previous = {
        "phase": "offline", "status": "complete", "gate": {"passed": True},
        "cases": synthetic_results("offline"), "preregistration_sha256": "prereg",
        "script_sha256": "script", "source_closure_reverified_at_end": True,
        "output_artifact_sha256": {artifact.name: experiment.sha256(artifact)},
    }
    path = tmp_path / "summary.json"
    experiment.write_json(path, previous)
    experiment.verify_previous_summary(path, "preflight", "prereg", "script")
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        experiment.verify_previous_summary(path, "preflight", "prereg", "script")
    previous["cases"][0]["pitch_bitwise_equal_archived49"] = False
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="did not pass"):
        experiment.verify_previous_summary(path, "preflight", "prereg", "script")


def test_sigterm_retains_prefix_and_unknown_inflight_counts(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    experiment.write_json(archive / "summary.json", {"cases": []})
    registration = tmp_path / "registration.json"
    experiment.write_json(registration, {"source_sha256": {}, "archive": str(archive)})
    monkeypatch.setattr(experiment, "PREREG", registration)
    previous = tmp_path / "previous.json"
    experiment.write_json(previous, {})
    monkeypatch.setattr(experiment, "verify_previous_summary", lambda *args: None)
    output = tmp_path / "estimator50_preflight_20261024_01"
    monkeypatch.setattr(sys, "argv", ["candidate50", "--phase", "preflight", "--output",
                                    str(output), "--previous-summary", str(previous)])

    def interrupted_case(case, destination):
        (destination / f"{experiment.case_name(case)}.csv").write_text("step\n1\n2\n")
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(experiment, "run_case", interrupted_case)
    old_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError):
        experiment.main()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["active_case"] == experiment.experiment_cases("preflight")[0]
    assert failure["active_case_logged_control_prefix"] == failure["control_step_lower_bound"] == 2
    assert failure["executed_control_steps"] is None
    assert failure["explicit_control_integration_substeps"] is None
    assert failure["count_boundary"]["total_native_calls"] is None
    assert not failure["counts_exact"] and not failure["resume_permitted"]
    assert signal.getsignal(signal.SIGTERM) == old_handler
    assert not (output / "summary.json").exists()
