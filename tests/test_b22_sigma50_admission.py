"""Admission rejection tests use no model, Torch or simulator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module50():
    spec = importlib.util.spec_from_file_location("_test_sigma50_admission", ROOT / "scripts/admit_b22_sigma50.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_admission_plan_has_no_training_or_native_budget():
    admission = module50()
    plan = admission.validate_plan50()
    assert plan["model_api_limits"]["predict"] == 2
    assert plan["minimum_complete_counts"]["predict_rows"] == 64
    assert plan["training_controls"] == plan["native_steps"] == plan["mujoco_constructions"] == 0
    assert plan["training_attempts_consumed"] == 0
    assert not any(name in sys.modules for name in ("torch", "mujoco", "stable_baselines3"))


def test_admission_plan_rejects_budget_drift(tmp_path, monkeypatch):
    admission = module50()
    plan = admission.validate_plan50()
    plan["model_api_limits"]["load"] = 3
    path = tmp_path / "changed_plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(admission, "PLAN", path)
    with pytest.raises(ValueError, match="preregistration differs"):
        admission.validate_plan50()


def test_zero_api_guard_refuses_before_underlying_call():
    admission = module50()
    called = []

    class Optimizer:
        def step(self):
            called.append("stepped")

    counter = admission.ZeroCounter50()
    counter.patch(Optimizer, "step", "optimizer_step")
    try:
        with pytest.raises(RuntimeError, match="zero API budget"):
            Optimizer().step()
        assert called == []
        assert counter.counts["optimizer_step"] == {"attempted": 0, "returned": 0, "rejected": 1}
    finally:
        counter.close()
    Optimizer().step()
    assert called == ["stepped"]


def test_simulator_import_guard_is_prefix_exact():
    admission = module50()
    blocker = admission.BlockImports50(admission.SIMULATORS)
    with pytest.raises(ImportError, match="forbidden dependency"):
        blocker.find_spec("mujoco._functions")
    assert blocker.find_spec("numpy") is None
    assert blocker.find_spec("mujoco_something_else") is None
    assert blocker.rejected == ["mujoco._functions"]


def test_exclusive_admission_output_retains_original(tmp_path, monkeypatch):
    admission = module50()
    monkeypatch.setattr(admission, "OUTPUT", tmp_path)
    target = tmp_path / "report.json"
    admission.write(target, {"status": "INCOMPLETE"})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        admission.write(target, {"status": "COMPLETE"})
    assert target.read_bytes() == original
    with pytest.raises(ValueError, match="escapes"):
        admission.write(tmp_path.parent / "unreserved.json", {})
