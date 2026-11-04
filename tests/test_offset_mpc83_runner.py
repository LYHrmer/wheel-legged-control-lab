"""Original absolute gates cannot be replaced by a weaker delayed baseline."""
import json

import pytest

from scripts import offset_mpc83 as runner


def cohort():
    return [{"seed": seed, "scenario": scenario, "profile": "benchmark", "arm": arm,
                 "completed": True, "success": 1, "steps": 400,
                 "height_rmse_mm": 100.0 if arm == "baseline" else 89.0,
                 "velocity_rmse_mps": 1.0, "pitch_rmse_deg": 1.0, "max_abs_pitch_deg": 1.0,
                 "normalized_effort_rms": 1.0, "wheel_force_rms_n": 1.0, "leg_force_rms_n": 1.0,
                 "solve_p95_ms": 1.0, "metric_crosscheck": {"passed": True},
                 "baseline82_replay_exact": arm == "baseline", "queue_checks_completed": 400,
                 "nominal_model_preserved": True, "observer_rows": 400, "terminal_observer_recorded": True}
            for seed in runner.SEEDS for scenario in runner.SCENARIOS for arm in runner.ARMS]


@pytest.fixture
def original(tmp_path, monkeypatch):
    spec = json.loads((runner.ROOT / runner.REG49).read_text())
    for path, data in ((runner.REG49, spec),
                       (runner.OLD49 + "/summary.json", {"episodes": [r for r in cohort() if r["arm"] == "baseline"]})):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))
    monkeypatch.setattr(runner, "ROOT", tmp_path)


def test_both_reference_gates_are_required(original):
    rows = cohort()
    assert runner.assess(rows)["accepted"]
    for row in rows:
        row["velocity_rmse_mps"] = 1.20 if row["arm"] == "baseline" else 1.10
    result = runner.assess(rows)
    assert result["incremental_Delay82"]["accepted"]
    assert not result["absolute_original49"]["accepted"]
    assert not result["accepted"]
    rows = cohort()
    for row in rows:
        if row["arm"] == "baseline":
            row["height_rmse_mm"] = 80.0
    result = runner.assess(rows)
    assert result["absolute_original49"]["accepted"]
    assert not result["incremental_Delay82"]["accepted"]
    assert not result["accepted"]


@pytest.mark.parametrize("change", ("partial", "duplicate", "missing", "terminal", "baseline", "queue"))
def test_incomplete_or_unverified_cohort_cannot_qualify(original, change):
    rows = cohort()
    if change == "partial":
        rows[0]["completed"] = False
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "missing":
        rows.pop()
    elif change == "terminal":
        rows[1]["terminal_observer_recorded"] = False
    elif change == "baseline":
        rows[0]["baseline82_replay_exact"] = False
    else:
        rows[1]["queue_checks_completed"] = 399
    result = runner.assess(rows)
    assert not result["accepted"]
    assert result["full_status"] == "NOT_RUN"


def test_original_mpc82_sources_cannot_be_rebound(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "required_inputs", lambda: ["frozen82.py"])
    frozen = tmp_path / runner.REG82
    frozen.parent.mkdir(parents=True)
    frozen.write_text(json.dumps({"source_sha256": {"frozen82.py": "original"}}))
    old = tmp_path / runner.REG49
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"source_sha256": {}}))
    monkeypatch.setattr(runner, "REG82_SHA", runner.sha(frozen))
    plan = dict(runner.SETTINGS, source_sha256={"frozen82.py": "rebound"})
    with pytest.raises(ValueError, match="frozen82 source closure"):
        runner.verify_plan(plan)


class FailingController:
    """A synthetic call can fail before or after entering its fake optimizer."""

    def __init__(self, failure_location):
        self.failure_location = failure_location
        self.calls = 0
        self.optimizer_entries = 0

    def reset(self, state=None):
        self.calls = self.optimizer_entries = 0

    def compute(self, state, command):
        self.calls += 1
        if self.calls == 2 and self.failure_location == "observer":
            raise ValueError("observer rejected input before optimizer")
        self.optimizer_entries += 1
        if self.calls == 2 and self.failure_location == "optimizer":
            raise InterruptedError("interrupted inside optimizer")
        return [0.0, 100.0]


@pytest.mark.parametrize("location,optimizer_entries", [("observer", 1), ("optimizer", 2)])
def test_failed_compute_cannot_be_counted_as_a_known_optimizer_invocation(location, optimizer_entries):
    underlying = FailingController(location)
    controller = runner.RecordedController(underlying)
    controller.reset()
    controller.compute([0.0] * 6, None)
    receipt = {"completed": True}
    runner.update_controller_accounting(receipt, controller, recorded_solve_samples=1)
    assert receipt["solve_calls"] == receipt["compute_successful_returns"] == 1
    with pytest.raises((ValueError, InterruptedError)):
        controller.compute([0.0] * 6, None)
    runner.update_controller_accounting(receipt, controller, recorded_solve_samples=1)
    assert underlying.optimizer_entries == optimizer_entries
    assert controller.compute_attempts == receipt["compute_attempts"] == 2
    assert receipt["compute_successful_returns"] == 1
    assert receipt["solve_calls"] is None
    assert receipt["solve_calls_completed_lower_bound"] == 1
    assert receipt["solve_counter_unresolved"] and not receipt["completed"]
    assert receipt["initial_solve_calls"] == 1


def test_unreturned_initial_compute_keeps_initial_solve_total_unknown():
    controller = runner.RecordedController(FailingController("observer"))
    controller.controller.calls = 1
    with pytest.raises(ValueError, match="before optimizer"):
        controller.compute([0.0] * 6, None)
    receipt = {"completed": False}
    runner.update_controller_accounting(receipt, controller)
    assert receipt["compute_attempts"] == 1 and receipt["compute_successful_returns"] == 0
    assert receipt["solve_calls"] is receipt["initial_solve_calls"] is None
    assert receipt["solve_calls_completed_lower_bound"] == 0
    assert receipt["initial_solve_calls_completed_lower_bound"] == 0


@pytest.mark.parametrize("bad_time,reason", [
    (.02, "simulation_clock_regressed"),
    (float("nan"), "nonfinite_or_invalid_clock"),
    (float("inf"), "nonfinite_or_invalid_clock"),
    (.041, "clock_not_on_native_step_grid"),
])
def test_invalid_clock_preserves_prior_integration_prefix_and_unknown_is_sticky(bad_time, reason):
    receipt = {"completed": True, "physics_steps": 0}
    runner.update_physics_accounting(receipt, .04, .002, completed_control_steps=2)
    assert receipt["physics_steps"] == receipt["physics_steps_lower_bound"] == 20
    runner.update_physics_accounting(receipt, bad_time, .002, completed_control_steps=2)
    assert receipt["physics_steps"] is None and receipt["physics_steps_lower_bound"] == 20
    assert receipt["last_resolved_simulation_time_s"] == .04
    assert receipt["physics_counter_unresolved"] and not receipt["completed"]
    assert receipt["physics_counter_failure"] == reason
    runner.update_physics_accounting(receipt, .06, .002, completed_control_steps=2)
    assert receipt["physics_steps"] is None and receipt["physics_steps_lower_bound"] == 20
    assert not receipt["completed"]


def test_valid_native_prefix_can_extend_beyond_a_saved_control_interval():
    receipt = {"completed": False, "physics_steps": 20, "physics_steps_lower_bound": 20}
    runner.update_physics_accounting(receipt, .05, .002, completed_control_steps=2)
    assert receipt["physics_steps"] == receipt["physics_steps_lower_bound"] == 25
    assert not receipt["physics_counter_unresolved"]
    assert not receipt["completed"]  # Partial native interval is not a complete episode.


def test_aggregate_counters_preserve_null_totals_and_add_only_known_lower_bounds():
    complete = {"steps": 8, "physics_steps": 80, "physics_steps_lower_bound": 80,
                    "solve_calls": 9, "solve_calls_completed_lower_bound": 9,
                    "compute_attempts": 9, "compute_successful_returns": 9, "initial_solve_calls": 1,
                    "initial_solve_calls_completed_lower_bound": 1}
    partial = {"steps": 2, "physics_steps": None, "physics_steps_lower_bound": 20,
                   "solve_calls": None, "solve_calls_completed_lower_bound": 3,
                   "compute_attempts": 4, "compute_successful_returns": 3, "initial_solve_calls": 1,
                   "initial_solve_calls_completed_lower_bound": 1}
    result = runner.aggregate_counters([complete, partial])
    assert result["control_steps"] == 10
    assert result["integration_steps"] is result["solve_calls"] is None
    assert result["integration_steps_lower_bound"] == 100
    assert result["solve_calls_completed_lower_bound"] == 12
    assert result["compute_attempts"] == 13 and result["compute_successful_returns"] == 12
    assert result["initial_solve_calls"] == 2


def test_failure_receipt_serializes_unknown_totals_without_losing_nonfinite_clock(tmp_path):
    receipt = {"completed": True, "physics_steps": 30, "physics_steps_lower_bound": 30}
    runner.update_physics_accounting(receipt, float("nan"), .002, completed_control_steps=3)
    receipt.update(solve_calls=None, solve_calls_completed_lower_bound=4)
    runner.persist_episode_receipt(tmp_path, receipt)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["physics_steps"] is saved["solve_calls"] is None
    assert saved["physics_steps_lower_bound"] == 30
    assert saved["solve_calls_completed_lower_bound"] == 4
    assert saved["last_simulation_time_s"] == "NaN"
    assert saved["status"] == "INCOMPLETE" and not saved["completed"]
