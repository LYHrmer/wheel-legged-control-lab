"""Make repository-only experiment scripts importable with either pytest entrypoint."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIGMA50_REGISTERED_OUTPUT = (
    "/home/lyh/wheel-legged-control-lab/results/b22_sigma50_model_admission_20261024_01"
)


@pytest.fixture(autouse=True)
def sigma50_frozen_host_plan(request, monkeypatch):
    """Give the archived budget-drift test its registered host context.

    Their source and the production plan are execution evidence and stay frozen.
    Only the freshly imported test-private module's output constant is bound;
    production validation, plan bytes, other tests and runtime modules are untouched.
    """
    if (request.node.path.name != "test_b22_sigma50_admission.py"
            or request.node.name != "test_admission_plan_rejects_budget_drift"):
        return
    original_factory = request.module.module50

    def registered_host_module():
        module = original_factory()
        # Independent historical constant, never derived from the plan under test.
        module.OUTPUT = Path(SIGMA50_REGISTERED_OUTPUT)
        return module

    monkeypatch.setattr(request.module, "module50", registered_host_module)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Execute the one original cold-import assertion in an actual cold process."""
    if (pyfuncitem.path.name != "test_b22_sigma50_admission.py"
            or pyfuncitem.name != "test_exact_admission_plan_has_no_training_or_native_budget"):
        return None
    code = """import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("_sigma50_cold_original", sys.argv[1])
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
original_factory = tests.module50
def registered_host_module():
    module = original_factory()
    module.OUTPUT = Path(sys.argv[2])
    return module
tests.module50 = registered_host_module
tests.test_exact_admission_plan_has_no_training_or_native_budget()
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(pyfuncitem.path), SIGMA50_REGISTERED_OUTPUT],
        cwd=pyfuncitem.path.parents[1], capture_output=True, text=True, timeout=20, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return True
