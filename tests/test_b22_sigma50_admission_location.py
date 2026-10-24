"""A relocated checkout must not make the production host-bound plan permissive."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_production_admission_rejects_foreign_output_after_valid_host(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/admit_b22_sigma50.py"
    spec = importlib.util.spec_from_file_location("_test_sigma50_location", source)
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    registered = Path(
        "/home/lyh/wheel-legged-control-lab/results/b22_sigma50_model_admission_20261024_01"
    )
    monkeypatch.setattr(admission, "OUTPUT", registered)
    assert admission.validate_plan50()["output"] == str(registered)
    monkeypatch.setattr(admission, "OUTPUT", tmp_path / "foreign_output")
    with pytest.raises(ValueError, match="preregistration differs"):
        admission.validate_plan50()


def test_cold_original_plan_preserves_parent_module_cache(monkeypatch):
    test_directory = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("_test_sigma50_cold_fixture", test_directory / "conftest.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    before = {}
    for name in ("torch", "mujoco", "stable_baselines3"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        before[name] = sys.modules[name]
    node = SimpleNamespace(
        path=test_directory / "test_b22_sigma50_admission.py",
        name="test_exact_admission_plan_has_no_training_or_native_budget",
    )
    assert fixture.pytest_pyfunc_call(node) is True
    assert all(sys.modules[name] is module for name, module in before.items())
