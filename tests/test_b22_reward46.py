"""Pure checks of the 46 tracking-shape transform."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import b22_reward46 as r
from b22_tracking_reward42 import apply_tracking_deficit42

BASE = {
    "tracking_vx": 2.0,
    "tracking_vy": 1.0,
    "tracking_yaw": 1.0,
    "height": -0.01,
    "attitude": -0.02,
    "actual_torque": -0.03,
    "action_change": -0.04,
    "leg_speed": -0.05,
    "termination": 0.0,
}


def terms(qx, qy):
    return {**BASE, "tracking_vx": 2 * math.exp(-qx), "tracking_yaw": math.exp(-qy)}


@pytest.mark.parametrize("alpha", [1, 8])
def test_beta_zero_is_bitwise_the_42_transform(alpha):
    original = terms(0.0123, 0.0456)
    assert r.apply_tracking_log_sse46(
        original, 0.0123, 0.0456, alpha=alpha, beta=0
    ) == apply_tracking_deficit42(original, alpha)


@pytest.mark.parametrize("alpha", [1, 8])
def test_beta_two_changes_only_the_two_tracking_terms(alpha):
    original = terms(0.04, 0.09)
    got = r.apply_tracking_log_sse46(original, 0.04, 0.09, alpha=alpha, beta=8)
    assert set(got) == set(BASE)
    for key in BASE:
        if key not in ("tracking_vx", "tracking_yaw"):
            assert got[key] == original[key]
    assert got["tracking_vx"] < apply_tracking_deficit42(original, alpha)["tracking_vx"]
    assert got["tracking_yaw"] < apply_tracking_deficit42(original, alpha)["tracking_yaw"]


def test_perfect_tracking_keeps_the_upper_bounds():
    got = r.apply_tracking_log_sse46(terms(0.0, 0.0), 0.0, 0.0, alpha=8, beta=8)
    assert got["tracking_vx"] == 2.0 and got["tracking_yaw"] == 1.0


def test_small_error_gradient_is_larger_than_the_exponential_transform():
    delta = 1e-4

    # central difference of tracking_vx in qx near a typical rough-terrain error
    def vx(qx):
        return r.apply_tracking_log_sse46(terms(qx, 0), qx, 0, alpha=8, beta=8)["tracking_vx"]

    def vx42(qx):
        return apply_tracking_deficit42(terms(qx, 0), 8)["tracking_vx"]

    q = 0.04
    slope46 = (vx(q + delta) - vx(q - delta)) / (2 * delta)
    slope42 = (vx42(q + delta) - vx42(q - delta)) / (2 * delta)
    assert slope46 < slope42  # a larger negative gradient
    assert slope46 > 3 * slope42  # but not an order-of-magnitude blow-up


def test_large_error_is_log_compressed_and_capped():
    q = 100.0
    original = {
        "tracking_vx": 2 * math.exp(-q),
        "tracking_yaw": math.exp(-q),
        **{k: v for k, v in BASE.items() if k not in ("tracking_vx", "tracking_yaw")},
    }
    got = r.apply_tracking_log_sse46(original, q, q, alpha=8, beta=8)
    penalty = 8 * math.log1p(r.CAP46)
    assert got["tracking_vx"] == pytest.approx(2.0 - 16.0 - penalty, abs=1e-9)
    assert got["tracking_yaw"] == pytest.approx(1.0 - 8.0 - penalty, abs=1e-9)
    assert penalty < 13.0  # bounded, same order as the -10 termination term


def test_normalized_errors_match_the_evaluation_scales():
    qx, qy = r.normalized_errors46(1.25, 0.48, 1.2, 0.4)
    assert qx == pytest.approx((0.05 / 0.25) ** 2)
    assert qy == pytest.approx((0.08 / 0.4) ** 2)


def test_inconsistent_original_terms_and_bad_inputs_are_refused():
    original = terms(0.04, 0.09)
    with pytest.raises(ValueError):
        r.apply_tracking_log_sse46(original, 0.05, 0.09)
    with pytest.raises(ValueError):
        r.apply_tracking_log_sse46(original, -0.1, 0.09)
    with pytest.raises(ValueError):
        r.apply_tracking_log_sse46(original, 0.04, 0.09, beta=3)
    with pytest.raises(TypeError):
        r.apply_tracking_log_sse46(original, True, 0.09)
    with pytest.raises(ValueError):
        r.apply_tracking_log_sse46(original, 0.04, float("nan"))
