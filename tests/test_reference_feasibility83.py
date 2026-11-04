"""Analytic torque conditions; no simulator or numerical package is imported."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

# Avoid the frozen package __init__, which eagerly imports NumPy/MuJoCo modules.
MODULE = Path(__file__).resolve().parents[1] / "src/wheel_legged_control/reference_feasibility83.py"
SPEC = importlib.util.spec_from_file_location("reference_feasibility83_standalone", MODULE)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
check_reference_torque = CHECKER.check_reference_torque


def checked(reference, **changes):
    parameters = {
        "dt": 0.5,
        "total_inertia": 2.0,
        "damping": 0.0,
        "coulomb": 0.0,
        "torque_limit": 20.0,
    }
    parameters.update(changes)
    return check_reference_torque(reference, **parameters)


def test_constant_reference_needs_only_next_speed_damping_and_pre_friction():
    result = checked([1.0, 1.0, 1.0], damping=2.0, coulomb=0.5, friction_smoothing=1.0)
    expected = 2.0 + 0.5 * math.tanh(1.0)
    assert [r["required_torque_nm"] for r in result["intervals"]] == pytest.approx([expected] * 2)
    assert all(r["inertial_torque_nm"] == 0.0 for r in result["intervals"])
    assert result["necessary_torque_condition"]
    assert result["closed_loop_tracking"] == "NOT_EVALUATED"
    assert result["rmse_lower_bound"] is None


def test_constant_acceleration_uses_explicit_last_sample_and_implicit_damping():
    # acceleration=2 rad/s^2; inertial torque=4; damping uses 1,2,3 (not 0,1,2).
    reference = [0.0, 1.0, 2.0, 3.0]
    result = checked(reference, damping=1.0)
    assert [r["required_torque_nm"] for r in result["intervals"]] == [5.0, 6.0, 7.0]
    assert result["sample_count"] == 4 and result["interval_count"] == 3
    assert reference == [0.0, 1.0, 2.0, 3.0]


def test_opposite_reference_reverses_torque_with_identical_violation_count():
    plus = checked([0.0, 1.0, 2.0], damping=1.0, coulomb=0.2, stribeck=0.3, torque_limit=5.0)
    minus = checked([0.0, -1.0, -2.0], damping=1.0, coulomb=0.2, stribeck=0.3, torque_limit=5.0)
    assert [r["required_torque_nm"] for r in minus["intervals"]] == pytest.approx(
        [-r["required_torque_nm"] for r in plus["intervals"]]
    )
    assert plus["violation_count"] == minus["violation_count"] == 1
    assert plus["violation_fraction"] == minus["violation_fraction"] == 0.5


def test_friction_is_preinterval_even_when_target_changes_sign():
    result = checked([1.0, -1.0], total_inertia=0.1, coulomb=1.0, friction_smoothing=1.0)
    assert result["intervals"][0]["preinterval_friction_torque_nm"] == math.tanh(1.0)
    assert result["intervals"][0]["required_torque_nm"] == pytest.approx(-0.4 + math.tanh(1.0))


def test_known_stribeck_coefficient_is_not_automatically_inferred():
    result = checked(
        [0.3, 0.3], coulomb=0.04, stribeck=0.08, stribeck_velocity=0.3, friction_smoothing=0.3
    )
    expected = (0.04 + 0.08 / math.e) * math.tanh(1.0)
    assert result["peak_abs_required_torque_nm"] == pytest.approx(expected)


def test_limit_equality_is_not_a_violation_and_no_clipping_is_applied():
    equal = checked([0.0, 1.0], total_inertia=1.0, torque_limit=2.0)
    above = checked([0.0, math.nextafter(1.0, math.inf)], total_inertia=1.0, torque_limit=2.0)
    assert equal["necessary_torque_condition"]
    assert not above["necessary_torque_condition"]
    assert above["intervals"][0]["required_torque_nm"] > 2.0
    assert above["parameters"]["torque_limit"] == 2.0
    assert not above["reference_or_torque_limit_modified"]


def test_fixed82_stress_reference_is_one_deterministic_reference_not_seed_replication():
    # Last target at t=6.0 is supplied from the SAME analytic command, not a
    # new physics transition or an extrapolation performed by the checker.
    dt = 0.002
    reference = [18.0 * math.sin(2.0 * math.pi * (k * dt)) for k in range(3001)]
    result = check_reference_torque(
        reference,
        dt=dt,
        total_inertia=0.5 * 0.5 * 0.087**2 + 0.017,
        damping=0.105,
        coulomb=0.043,
        stribeck=0.08,
        torque_limit=2.0,
    )
    assert result["sample_count"] == 3001 and result["interval_count"] == 3000
    assert result["violation_count"] == 1512
    assert result["violation_fraction"] == 0.504
    assert result["peak_abs_required_torque_nm"] == pytest.approx(2.90448655515, rel=0.0, abs=1e-10)
    assert not result["necessary_torque_condition"]


@pytest.mark.parametrize(
    "reference", ([], [0.0], "01", [0.0, math.nan], [math.inf, 0.0], [False, 1.0])
)
def test_bad_or_incomplete_reference_is_rejected(reference):
    with pytest.raises(ValueError):
        checked(reference)


@pytest.mark.parametrize(
    "name", ("dt", "total_inertia", "torque_limit", "friction_smoothing", "stribeck_velocity")
)
@pytest.mark.parametrize("value", (0.0, -1.0, math.nan, math.inf, True))
def test_positive_parameters_are_validated(name, value):
    with pytest.raises(ValueError):
        checked([0.0, 1.0], **{name: value})


@pytest.mark.parametrize("name", ("damping", "coulomb", "stribeck"))
@pytest.mark.parametrize("value", (-1.0, math.nan, math.inf, True))
def test_nonnegative_parameters_are_validated(name, value):
    with pytest.raises(ValueError):
        checked([0.0, 1.0], **{name: value})


@pytest.mark.parametrize(
    "reference, parameters",
    (
        ([-1e308, 1e308], {}),  # reference subtraction
        ([0.0, 1e308], {"total_inertia": 1e308}),  # inertial product
        ([0.0, 1.0], {"dt": 5e-324}),  # inertial division
        ([1e308, 1e308], {"damping": 1e308}),  # implicit damping
        ([1e308, 1e308], {"coulomb": 1.0}),  # smoothing ratio
        ([1e200, 1e200], {"stribeck": 1.0}),  # squared Stribeck ratio
        ([0.0, 1.0], {"total_inertia": 5e307, "damping": 1e308}),  # torque sum
    ),
)
def test_finite_inputs_with_overflow_cannot_pass(reference, parameters):
    with pytest.raises(ValueError, match="overflow|nonfinite"):
        checked(reference, **parameters)


def test_out_of_binary64_integer_rejected_and_finite_extreme_zero_load_is_valid():
    with pytest.raises(ValueError, match="binary64"):
        checked([0, 10**400])
    result = checked([1e308, 1e308])
    assert result["peak_abs_required_torque_nm"] == 0.0
    assert result["necessary_torque_condition"]
