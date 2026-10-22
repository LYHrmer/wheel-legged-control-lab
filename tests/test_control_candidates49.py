"""Boundary checks for one-shot comparison admission and feedforward injection."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from scripts import evaluate_wheel_control as legacy
from scripts.control_candidates49 import (
    admit_full,
    decision,
    scaled_feedforward_controller,
    sha256,
    stream_pi_rows,
)
from wheel_legged_control.actuator_bench import ActuatorParameters, BenchConfig
from wheel_legged_control.wheel_control import WheelControlConfig


def pi_cohort():
    spec = {"preflight_seeds": [1, 2, 3], "scenarios": ["matched", "mismatch"],
            "profiles": ["tracking", "reversal", "stress"]}
    rows = [{"seed": seed, "scenario": scenario, "profile": "stress", "arm": arm,
             "completed": True, "speed_guard_triggered": False,
             "velocity_rmse_rad_s": 1.0,
             "saturation_fraction": 0.6 if arm == "baseline" else 0.5}
            for seed in spec["preflight_seeds"] for scenario in spec["scenarios"]
            for arm in ("baseline", "candidate")]
    return spec, rows


def test_partial_duplicate_or_missing_episodes_cannot_qualify():
    spec, rows = pi_cohort()
    assert decision(rows, "pi", "preflight", spec)["accepted"]
    for invalid in (rows[:-1], rows + [rows[0]], [{**rows[0], "completed": False}, *rows[1:]]):
        result = decision(invalid, "pi", "preflight", spec)
        assert not result["accepted"]
        assert result["gates"] == {}


def test_saturation_benefit_does_not_hide_tracking_regression_or_protection():
    spec, rows = pi_cohort()
    bad_tracking = [{**row, "velocity_rmse_rad_s": 1.06} if row["arm"] == "candidate"
                    else row for row in rows]
    assert not decision(bad_tracking, "pi", "preflight", spec)["accepted"]
    rows[1]["speed_guard_triggered"] = True
    assert not decision(rows, "pi", "preflight", spec)["accepted"]


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_scaled_feedforward_retains_conditional_integration_and_limit(direction):
    # Scaling a 1.2 Nm feedforward to 0.9 Nm permits a 0.1 Nm integral step;
    # clipping the old controller's output after compute would incorrectly freeze it.
    config = WheelControlConfig(kp=0.0, ki=1.0, dt=0.1, torque_limit_nm=1.0)
    parameters = ActuatorParameters(armature=0.0, damping=1.2, coulomb_friction=0.0)
    baseline = scaled_feedforward_controller(config, parameters, 1.0)
    candidate = scaled_feedforward_controller(config, parameters, 0.75)
    old = baseline.compute(direction, 0.0, 0.0)
    new = candidate.compute(direction, 0.0, 0.0)
    assert old.integration_frozen
    assert not new.integration_frozen
    assert new.integral_torque_nm == pytest.approx(direction * 0.1)
    assert new.command_torque_nm == pytest.approx(direction)
    assert parameters.damping == 1.2


def test_full_refuses_failed_or_tampered_preflight(tmp_path):
    summary = {"candidate": "pi", "phase": "preflight", "registration_sha256": "registered",
               "status": "complete", "decision": {"accepted": False}}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    (tmp_path / "manifest.json").write_text(json.dumps({"summary.json": sha256(path)}))
    with pytest.raises(ValueError, match="rejected"):
        admit_full(tmp_path, "pi", "registered")
    summary["decision"]["accepted"] = True
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="artifact changed"):
        admit_full(tmp_path, "pi", "registered")
    with pytest.raises(ValueError, match="requires"):
        admit_full(None, "pi", "registered")


class FakeBench:
    """Deterministic test double; no MuJoCo model or physics calls."""

    stop_after = None

    def __init__(self, config, truth, *, stribeck_friction_nm=0.0):
        self.config = config
        self.reset()

    def reset(self):
        self.state = np.zeros(2)
        self.applied_torque_nm = 0.0
        self.steps = 0

    def step(self, command):
        if self.stop_after is not None and self.steps == self.stop_after:
            raise InterruptedError("simulated interruption")
        self.applied_torque_nm = command
        self.state = self.state + np.array([self.config.dt * self.state[1], command * 0.01])
        self.steps += 1
        return self.state


@pytest.mark.parametrize("profile", ["tracking", "reversal", "stress"])
@pytest.mark.parametrize("scale", [1.0, 0.75])
def test_streamed_rows_match_frozen_evaluator_sampling_and_noise(
    monkeypatch, tmp_path, profile, scale,
):
    monkeypatch.setattr(legacy, "ActuatorBench", FakeBench)
    config, truth = BenchConfig(), ActuatorParameters()
    controller = scaled_feedforward_controller(WheelControlConfig(), truth, scale)
    kwargs = {"profile": profile, "duration": 0.02, "seed": 490101,
              "noise_scale": 1.0, "stribeck_friction_nm": 0.08}
    expected = legacy.run_closed_loop(config, truth, controller, **kwargs)
    actual = stream_pi_rows(config, truth, controller, destination=tmp_path, **kwargs)
    assert actual == expected
    with (tmp_path / "trajectory.csv").open(newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == len(expected) == 10
    assert [float(r["actual_velocity_rad_s"]) for r in saved] == [
        r["actual_velocity_rad_s"] for r in expected
    ]


def test_interrupted_pi_keeps_completed_transitions(monkeypatch, tmp_path):
    monkeypatch.setattr(legacy, "ActuatorBench", FakeBench)
    monkeypatch.setattr(FakeBench, "stop_after", 2)
    config, truth = BenchConfig(), ActuatorParameters()
    controller = scaled_feedforward_controller(WheelControlConfig(), truth, 0.75)
    with pytest.raises(InterruptedError, match="simulated interruption"):
        stream_pi_rows(config, truth, controller, profile="stress", duration=0.02,
                       seed=490101, noise_scale=1.0, stribeck_friction_nm=0.0,
                       destination=tmp_path)
    with (tmp_path / "trajectory.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
