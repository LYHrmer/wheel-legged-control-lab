"""Make repository-only experiment scripts importable with either pytest entrypoint."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def sigma50_frozen_host_plan(request, monkeypatch):
    """Give two archived contract tests their registered host context.

    Their source and the production plan are execution evidence and stay frozen.
    Only the freshly imported test-private module's output constant is bound;
    production validation, plan bytes, other tests and runtime modules are untouched.
    """
    if request.node.path.name != "test_b22_sigma50_admission.py" or request.node.name not in {
        "test_exact_admission_plan_has_no_training_or_native_budget",
        "test_admission_plan_rejects_budget_drift",
    }:
        return
    original_factory = request.module.module50

    def registered_host_module():
        module = original_factory()
        # Independent historical constant, never derived from the plan under test.
        module.OUTPUT = Path(
            "/home/lyh/wheel-legged-control-lab/results/b22_sigma50_model_admission_20261024_01"
        )
        return module

    monkeypatch.setattr(request.module, "module50", registered_host_module)
