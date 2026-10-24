"""A relocated checkout must not make the production host-bound plan permissive."""

import importlib.util
from pathlib import Path

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
